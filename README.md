# RadioDiff-FS: Physics-Informed Manifold Alignment in Few-Shot Diffusion Models for High-Fidelity Radio Map Construction

---
### Welcome to the RadioDiff family

Base BackBone, Paper Link: [RadioDiff](https://ieeexplore.ieee.org/document/10764739), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff), **IEEE TCCN**, 2025

PINN Enhanced with Helmholtz Equation, Paper Link: [RadioDiff-$k^2$](https://ieeexplore.ieee.org/document/11278649), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff-k), **IEEE JSAC**, 2026

Efficiency Enhanced RadioDiff, Paper Link: [RadioDiff-Turbo](https://ieeexplore.ieee.org/abstract/document/11152929/), **IEEE INFOCOM wksp**, 2025

Dynamic Environment or BS Location Change, Paper Link: [RadioDiff-Flux](https://ieeexplore.ieee.org/document/11282987/), **IEEE TCCN**, 2026

Few-Shot Learning, Paper Link: [RadioDiff-FS](https://arxiv.org/abs/2603.18865), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff-FS/blob/main/README.md)

Indoor RM Construction with Physical Information, Paper Link: [iRadioDiff](https://arxiv.org/abs/2511.20015), Code Link: [GitHub](https://github.com/UNIC-Lab/iRadioDiff), **IEEE ICC**, 2026

3D RM with DataSet, Paper Link: [RadioDiff-3D](https://ieeexplore.ieee.org/document/11083758), Code Link: [GitHub](https://github.com/UNIC-Lab/UrbanRadio3D), **IEEE TNSE**, 2025

Sparse Measurement for RM ISAC, Paper Link: [RadioDiff-Inverse](https://arxiv.org/abs/2504.14298), **IEEE TWC**, 2026

Sparse Measurement for NLoS Localization, Paper Link: [RadioDiff-Loc](https://www.arxiv.org/abs/2509.01875)

For more RM information, please visit the repo of [Awesome-Radio-Map-Categorized](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized)

---

##  Before Starting

1. install torch
~~~
conda create -n radiodiff python=3.9
conda avtivate radiodiff
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~
3. prepare accelerate config.
~~~
accelerate config # HOW MANY GPUs YOU WANG TO USE.
~~~

##  Prepare Data

##### We used the [RadioMapSeer](https://radiomapseer.github.io/) dataset for model training and testing.

- The data structure should look like:

```commandline
|-- $RadioMapSeer
|   |-- gain
|   |-- |-- carsDPM
|   |-- |-- |-- XXX_XX.PNG
|   |-- |-- |-- XXX_XX.PNG
|   ...
|   |-- png
|   |-- |-- buildings_complete
|   |-- |-- |-- XXX_XX.PNG
|   |-- |-- |-- XXX_XX.PNG
|	...
```
## :tada: Training
1. train the first stage model (AutoEncoder):
~~~
accelerate launch train_vae.py --cfg ./configs/first_radio.yaml
~~~
2. train latent diffusion-edge model:
~~~
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train.yaml
~~~
3. fine-tune the pretrained model on IRT4 few-shot data:
~~~
accelerate launch train_cond_ldm_finetune.py --cfg ./configs/radio_train_irt4_finetune.yaml
~~~

## V. Inference.
~~~
python sample_cond_ldm.py --cfg ./configs/radio_test_finetune.yaml
~~~

