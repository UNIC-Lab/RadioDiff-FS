import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class RadioUNet_c(Dataset):
    def __init__(
        self,
        maps_inds=np.zeros(1),
        phase="train",
        ind1=0,
        ind2=0,
        dir_dataset="./data/RadioMapSeer/",
        data_dir=None,
        numTx=80,
        thresh=0.05,
        simulation="DPM",
        carsSimul="no",
        carsInput="no",
        IRT2maxW=1,
        cityMap="complete",
        missing=1,
        transform=transforms.ToTensor(),
        shuffle_maps=False,
    ):
        self.transform_GY = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform_compose = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        if data_dir is not None:
            dir_dataset = data_dir

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            if shuffle_maps:
                np.random.seed(42)
                np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 501
            self.ind2 = 699
        else:
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh
        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        self.IRT2maxW = IRT2maxW
        self.cityMap = cityMap
        self.missing = missing
        self.transform = transform

        if simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = os.path.join(self.dir_dataset, "gain/DPM/")
            else:
                self.dir_gain = os.path.join(self.dir_dataset, "gain/carsDPM/")
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = os.path.join(self.dir_dataset, "gain/IRT2/")
            else:
                self.dir_gain = os.path.join(self.dir_dataset, "gain/carsIRT2/")
        elif simulation == "IRT4":
            if carsSimul == "no":
                self.dir_gain = os.path.join(self.dir_dataset, "gain/IRT4/")
            else:
                self.dir_gain = os.path.join(self.dir_dataset, "gain/carsIRT4/")
        elif simulation == "rand":
            if carsSimul == "no":
                self.dir_gainDPM = os.path.join(self.dir_dataset, "gain/DPM/")
                self.dir_gainIRT2 = os.path.join(self.dir_dataset, "gain/IRT2/")
            else:
                self.dir_gainDPM = os.path.join(self.dir_dataset, "gain/carsDPM/")
                self.dir_gainIRT2 = os.path.join(self.dir_dataset, "gain/carsIRT2/")

        if cityMap == "complete":
            self.dir_buildings = os.path.join(self.dir_dataset, "png/buildings_complete/")
        else:
            self.dir_buildings = os.path.join(self.dir_dataset, "png/buildings_missing")

        self.dir_Tx = os.path.join(self.dir_dataset, "png/antennas/")
        if carsInput != "no":
            self.dir_cars = os.path.join(self.dir_dataset, "png/cars/")

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1

    def __getitem__(self, idx):
        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        name1 = str(dataset_map_ind) + ".png"
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(
                self.dir_buildings + str(self.missing) + "/" + str(version) + "/",
                name1,
            )
        image_buildings = np.asarray(io.imread(img_name_buildings))

        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 255
        else:
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            w = np.random.uniform(0, self.IRT2maxW)
            image_gain = (
                w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256
                + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256
            )

        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx, image_buildings], axis=2)
        else:
            image_buildings = image_buildings / 256
            image_Tx = image_Tx / 256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, image_cars], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)

        out = {}
        out["image"] = self.transform_compose(image_gain)
        out["cond"] = self.transform_GY(inputs)
        out["img_name"] = name2
        return out


class RadioUNet_c_sprseIRT4(Dataset):
    def __init__(
        self,
        maps_inds=np.zeros(1),
        phase="train",
        ind1=0,
        ind2=0,
        dir_dataset="RadioMapSeer/",
        numTx=2,
        thresh=0.2,
        simulation="IRT4",
        carsSimul="no",
        carsInput="no",
        cityMap="complete",
        missing=1,
        num_samples=300,
        transform=transforms.ToTensor(),
        shuffle_maps=False,
    ):
        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            if shuffle_maps:
                np.random.seed(42)
                np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 501
            self.ind2 = 699
        else:
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh
        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        self.cityMap = cityMap
        self.missing = missing
        self.transform = transform
        self.num_samples = num_samples

        self.transform_GY = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform_compose = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        if simulation == "IRT4":
            if carsSimul == "no":
                self.dir_gain = os.path.join(self.dir_dataset, "gain/IRT4/")
            else:
                self.dir_gain = os.path.join(self.dir_dataset, "gain/carsIRT4/")
        elif simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = os.path.join(self.dir_dataset, "gain/DPM/")
            else:
                self.dir_gain = os.path.join(self.dir_dataset, "gain/carsDPM/")
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = os.path.join(self.dir_dataset, "gain/IRT2/")
            else:
                self.dir_gain = os.path.join(self.dir_dataset, "gain/carsIRT2/")

        if cityMap == "complete":
            self.dir_buildings = os.path.join(self.dir_dataset, "png/buildings_complete/")
        else:
            self.dir_buildings = os.path.join(self.dir_dataset, "png/buildings_missing")

        self.dir_Tx = os.path.join(self.dir_dataset, "png/antennas/")
        if carsInput != "no":
            self.dir_cars = os.path.join(self.dir_dataset, "png/cars/")

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):
        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        name1 = str(dataset_map_ind) + ".png"
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(
                self.dir_buildings + str(self.missing) + "/" + str(version) + "/",
                name1,
            )
        image_buildings = np.asarray(io.imread(img_name_buildings))

        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 256
        else:
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            w = np.random.uniform(0, self.IRT2maxW)
            image_gain = (
                w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256
                + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256
            )

        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        image_samples = np.zeros((self.width, self.height))
        seed_map = np.sum(image_buildings)
        np.random.seed(seed_map)
        x_samples = np.random.randint(0, 255, size=self.num_samples)
        y_samples = np.random.randint(0, 255, size=self.num_samples)
        image_samples[x_samples, y_samples] = 1

        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx, image_buildings], axis=2)
        else:
            image_buildings = image_buildings / 256
            image_Tx = image_Tx / 256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, image_cars], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_samples = self.transform(image_samples).type(torch.float32)

        out = {}
        out["image"] = self.transform_compose(image_gain)
        out["cond"] = self.transform_GY(inputs)
        out["img_name"] = name2
        out["ori_mask"] = image_samples
        return out
