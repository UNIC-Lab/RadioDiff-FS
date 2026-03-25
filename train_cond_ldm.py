import yaml
import argparse
import math
import torch
from lib import loaders
import torch.nn as nn
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.ema import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
import torch.nn.functional as F
from lib.modules import convrelu


def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def main(args):
    cfg = CfgNode(args.cfg)
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )

    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    )
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=model_cfg.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    data_cfg = cfg.data

    if data_cfg['name'] == 'edge':
        dataset = EdgeDataset(
            data_root=data_cfg.img_folder,
            image_size=model_cfg.image_size,
            augment_horizontal_flip=data_cfg.augment_horizontal_flip,
            cfg=data_cfg
        )
    elif data_cfg['name'] == 'radio':
        dataset = loaders.RadioUNet_c(
            phase=data_cfg.get('phase', 'train'),
            dir_dataset=data_cfg.get('data_dir', '/home/zxguo/RadioDiff_1/data/'),
            simulation=data_cfg.get('simulation', 'IRT4'),
            carsSimul=data_cfg.get('carsSimul', 'no'),
            carsInput=data_cfg.get('carsInput', 'no'),
            numTx=data_cfg.get('numTx', 2),
            thresh=data_cfg.get('thresh', 0.2),
            shuffle_maps=data_cfg.get('shuffle_maps', False),
        )
    else:
        raise NotImplementedError
    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))
    train_cfg = cfg.trainer
    trainer = Trainer(
        ldm, dl, train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg.lr, train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every, results_folder=train_cfg.results_folder,
        amp=train_cfg.amp, fp16=train_cfg.fp16, log_freq=train_cfg.log_freq, cfg=cfg,
        resume_milestone=train_cfg.resume_milestone,
        train_wd=train_cfg.get('weight_decay', 1e-4)
    )
    if train_cfg.test_before:
        if trainer.accelerator.is_main_process:
            with torch.no_grad():
                for datatmp in dl:
                    break
                if isinstance(trainer.model, nn.parallel.DistributedDataParallel):
                    all_images, *_ = trainer.model.module.sample(batch_size=datatmp['cond'].shape[0],
                                                  cond=datatmp['cond'].to(trainer.accelerator.device),
                                                  mask=datatmp['ori_mask'].to(trainer.accelerator.device) if 'ori_mask' in datatmp else None)
                elif isinstance(trainer.model, nn.Module):
                    all_images, *_ = trainer.model.sample(batch_size=datatmp['cond'].shape[0],
                                                  cond=datatmp['cond'].to(trainer.accelerator.device),
                                                  mask=datatmp['ori_mask'].to(trainer.accelerator.device) if 'ori_mask' in datatmp else None)

            nrow = 2 ** math.floor(math.log2(math.sqrt(data_cfg.batch_size)))
            tv.utils.save_image(all_images, str(trainer.results_folder / f'sample-{train_cfg.resume_milestone}_{model_cfg.sampling_timesteps}.png'), nrow=nrow)
            torch.cuda.empty_cache()
    trainer.train()
    pass


class Trainer(object):
    def __init__(
            self,
            model,
            data_loader,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_wd=1e-4,
            train_num_steps=100000,
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_freq=20,
            resume_milestone=0,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            kwargs_handlers=[ddp_handler],
        )
        self.enable_resume = cfg.trainer.get('enable_resume', False)
        self.accelerator.native_amp = amp

        self.model = model
        self.cfg = cfg

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        self.image_size = model.image_size



        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=train_lr, weight_decay=train_wd)
        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.96, cfg.trainer.min_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.ema = EMA(model, ema_model=None, beta=0.999,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)


        self.step = 0


        self.model, self.opt, self.lr_scheduler = \
            self.accelerator.prepare(self.model, self.opt, self.lr_scheduler)
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.writer = SummaryWriter(results_folder)
        self.results_folder = Path(results_folder)
        resume_file = str(self.results_folder / f'model-{resume_milestone}.pt')
        if os.path.isfile(resume_file):
            self.load(resume_milestone)

        self.use_directional_loss = self.cfg.trainer.get('use_directional_loss', False)
        if self.use_directional_loss:
            device = self.accelerator.device
            self.feature_net_src = self.build_feature_encoder(in_channels=3).to(device)
            self.feature_net_tgt = self.build_feature_encoder(in_channels=1).to(device)
            for p in self.feature_net_src.parameters():
                p.requires_grad = False
            for p in self.feature_net_tgt.parameters():
                p.requires_grad = False
            self.feature_net_src.eval()
            self.feature_net_tgt.eval()
            self.feature_dir = self.calc_feature_direction(num_batches=self.cfg.trainer.get('dir_estimate_batches', 4))

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        if self.enable_resume:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt': self.opt.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        else:
            data = {
                'model': self.accelerator.get_state_dict(self.model),
                'ema': self.ema.state_dict(),
            }
            torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        assert self.enable_resume; 'resume is available only if self.enable_resume is True !'
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.lr_scheduler.load_state_dict(data['lr_scheduler'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def build_feature_encoder(self, in_channels=1):
        class RadioUNetEncoder(nn.Module):
            def __init__(self, inputs):
                super().__init__()
                if inputs <= 3:
                    self.layer00 = convrelu(inputs, 6, 3, 1, 1)
                    self.layer0 = convrelu(6, 40, 5, 2, 2)
                else:
                    self.layer00 = convrelu(inputs, 10, 3, 1, 1)
                    self.layer0 = convrelu(10, 40, 5, 2, 2)
                self.layer1 = convrelu(40, 50, 5, 2, 2)
                self.layer10 = convrelu(50, 60, 5, 2, 1)
                self.layer2 = convrelu(60, 100, 5, 2, 2)
                self.layer20 = convrelu(100, 100, 3, 1, 1)
                self.layer3 = convrelu(100, 150, 5, 2, 2)
                self.layer4 = convrelu(150, 300, 5, 2, 2)
                self.layer5 = convrelu(300, 500, 5, 2, 2)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()

            def forward(self, x):
                x = self.layer00(x)
                x = self.layer0(x)
                x = self.layer1(x)
                x = self.layer10(x)
                x = self.layer2(x)
                x = self.layer20(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                x = self.pool(x)
                x = self.flatten(x)
                return x

        return RadioUNetEncoder(in_channels)

    @torch.no_grad()
    def calc_feature_direction(self, num_batches=4):
        src_features = []
        tgt_features = []
        device = self.accelerator.device
        iters = max(1, int(num_batches))
        for _ in range(iters):
            batch = next(self.dl)
            cond = batch['cond'].to(device)
            img = batch['image'].to(device)
            src_feat = self.feature_net_src(cond)
            tgt_feat = self.feature_net_tgt(img)
            src_features.append(src_feat)
            tgt_features.append(tgt_feat)
        src_mean = torch.cat(src_features, dim=0).mean(dim=0)
        tgt_mean = torch.cat(tgt_features, dim=0).mean(dim=0)
        return (tgt_mean - src_mean).detach()

    def directional_loss(self, gen_images, cond_images):
        gen_features = self.feature_net_tgt(gen_images)
        with torch.no_grad():
            src_features = self.feature_net_src(cond_images)
        target_features = src_features + self.feature_dir
        loss_per_sample = F.mse_loss(gen_features, target_features, reduction='none')
        if loss_per_sample.dim() > 1:
            loss_per_sample = loss_per_sample.mean(dim=1)
        return loss_per_sample

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                total_loss_dict = {'loss_simple': 0., 'loss_vlb': 0., 'total_loss': 0., 'lr': 5e-5}
                for ga_ind in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key].to(device)
                    if self.step == 0 and ga_ind == 0:
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            self.model.module.on_train_batch_start(batch)
                        else:
                            self.model.on_train_batch_start(batch)

                    with self.accelerator.autocast():
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            loss, log_dict = self.model.module.training_step(batch)
                        else:
                            loss, log_dict = self.model.training_step(batch)

                        if self.use_directional_loss:
                            if (self.step % 2) == 0:
                                with torch.no_grad():
                                    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                        sampled = self.model.module.sample(
                                            batch_size=batch['cond'].shape[0],
                                            cond=batch['cond'].to(self.accelerator.device)
                                        )
                                    else:
                                        sampled = self.model.sample(
                                            batch_size=batch['cond'].shape[0],
                                            cond=batch['cond'].to(self.accelerator.device)
                                        )
                                if isinstance(sampled, (tuple, list)):
                                    gen_images = sampled[0]
                                else:
                                    gen_images = sampled
                                dir_loss = self.directional_loss(gen_images, batch['cond'].to(self.accelerator.device)).to(loss.device)
                                t_hat = torch.rand(dir_loss.shape[0], device=dir_loss.device)
                                alpha = torch.pow(torch.tensor(20.0, device=dir_loss.device), t_hat)
                                dir_weight = self.cfg.trainer.get('directional_weight', 0.1)
                                loss = loss + (dir_weight * alpha * dir_loss).mean()
                            else:
                                lowt_weight = self.cfg.trainer.get('lowt_weight', 1.0)
                                phase_threshold = self.cfg.trainer.get('phase_threshold', 0.5)
                                if lowt_weight != 1.0 and phase_threshold > 0:
                                    expected_scale = (1 - phase_threshold) + phase_threshold * lowt_weight
                                    loss = loss * expected_scale

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        loss_simple = log_dict["train/loss_simple"].item() / self.gradient_accumulate_every
                        loss_vlb = log_dict["train/loss_vlb"].item() / self.gradient_accumulate_every
                        total_loss_dict['loss_simple'] += loss_simple
                        total_loss_dict['loss_vlb'] += loss_vlb
                        total_loss_dict['total_loss'] += total_loss

                    self.accelerator.backward(loss)
                total_loss_dict['lr'] = self.opt.param_groups[0]['lr']
                describtions = dict2str(total_loss_dict)
                describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                if accelerator.is_main_process:
                    pbar.desc = describtions

                if self.step % self.log_freq == 0:
                    if accelerator.is_main_process:
                        self.logger.info(describtions)

                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.lr_scheduler.step()
                if accelerator.is_main_process:
                    self.writer.add_scalar('Learning_Rate', self.opt.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('total_loss', total_loss, self.step)
                    self.writer.add_scalar('loss_simple', loss_simple, self.step)
                    self.writer.add_scalar('loss_vlb', loss_vlb, self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        self.model.eval()

                        with torch.no_grad():
                            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                all_images, *_ = self.model.module.sample(batch_size=batch['cond'].shape[0],
                                                  cond=batch['cond'],
                                                  mask=batch['ori_mask'] if 'ori_mask' in batch else None)
                            elif isinstance(self.model, nn.Module):
                                all_images, *_ = self.model.sample(batch_size=batch['cond'].shape[0],
                                                  cond=batch['cond'],
                                                  mask=batch['ori_mask'] if 'ori_mask' in batch else None)

                        nrow = 2 ** math.floor(math.log2(math.sqrt(batch['cond'].shape[0])))
                        tv.utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=nrow)
                        self.model.train()
                accelerator.wait_for_everyone()
                pbar.update(1)

        accelerator.print('training complete')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass


