import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from medmnist_dataset import get_medmnist_dataset, DataFlag
from typing import List, Literal, Optional, Tuple
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision.transforms as T
from collections import Counter
from random import choice
from numpy import linspace
import torch
import pickle
import os


class DataModule2D(pl.LightningDataModule):
    def __init__(self,
                 data_flag: DataFlag,
                 img_size: Literal[28, 64, 128, 224] = 28,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 selected_train_transforms: List[str] = ["ToImage", "DoDtype", "Normalize", "RandomResizedCrop"],
                 mmap_mode: Optional[Literal["r"]] = None):
        super().__init__()
        self.data_flag = data_flag
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.selected_train_transforms = selected_train_transforms
        self.mmap_mode = mmap_mode

    
    def prepare_data(self):
        get_medmnist_dataset(data_flag=self.data_flag, size=self.img_size, download=True)

        # Computing MEAN and STD on the whole training set.
        if not os.path.exists(f"fitted_factors/{self.data_flag.value}.pkl"):
            self._compute_and_save_mean_std()

        # Computing class weights on the whole training set.
        if not os.path.exists(f'/home/dzban112/MedMNIST/fitted_factors/class_weights/{self.data_flag.value}.pt'):
            self._calculate_class_weights(self.data_flag)


    @rank_zero_only
    def _compute_and_save_mean_std(self, batch_size=32):
        dataset = get_medmnist_dataset(mode="train",
                                       data_flag=self.data_flag,
                                       size=self.img_size,
                                       data_transform=v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)]),
                                       download=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.n_channels = dataset.info['n_channels']

        channel_sum = torch.zeros(self.n_channels)
        channel_sum_of_squares = torch.zeros(self.n_channels)
        pixel_count = 0
        for images, _ in loader:
            channel_sum += torch.sum(images, dim=(0, 2, 3))
            channel_sum_of_squares += torch.sum(images ** 2, dim=(0, 2, 3))
            pixel_count += images.size(0) * images.size(2) * images.size(3)

        channel_mean = channel_sum / pixel_count
        channel_std = torch.sqrt((channel_sum_of_squares / pixel_count) - (channel_mean ** 2))

        if not os.path.exists("fitted_factors/"):
            os.makedirs("fitted_factors/")
        with open(f"fitted_factors/{self.data_flag.value}.pkl", "wb") as file:
            pickle.dump((channel_mean, channel_std), file)
    
    
    @rank_zero_only
    def _calculate_class_weights(self, data_flag:DataFlag):
        """
        Calculate class weights for a specified dataset.
        """
        try:
            weights = torch.load(f'/home/dzban112/MedMNIST/fitted_factors/class_weights/{data_flag.value}.pt')
        except FileNotFoundError:
            print("Computing class weights on a training set.")
            dataset = get_medmnist_dataset(mode="train",
                                           data_flag=data_flag,
                                           size=28,
                                           data_transform=None,
                                           download=True)
            targets = [dataset[i][1][0] for i in range(len(dataset))]
            class_counts = Counter(targets)
            total_samples = len(targets)
            
            # Inverse proportionality to class frequency
            class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
            
            # Convert class_weights to tensor
            weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float)
            torch.save(weights, f'/home/dzban112/MedMNIST/fitted_factors/class_weights/{data_flag.value}.pt')

    
    def setup(self, stage=None):
        try:
            with open(f"fitted_factors/{self.data_flag.value}.pkl", "rb") as file:
                self.mean_std = pickle.load(file)
        except FileNotFoundError:
            raise RuntimeError("File with MEAN and STD not found.")

        if stage == 'fit' or stage is None:
            train_transforms, eval_transforms = self._get_transforms(train_transforms=self.selected_train_transforms)
            self.train_ds = get_medmnist_dataset(mode="train",
                                                 data_flag=self.data_flag,
                                                 data_transform=train_transforms,
                                                 size=self.img_size,
                                                 mmap_mode=self.mmap_mode)
            self.val_ds = get_medmnist_dataset(mode="val",
                                               data_flag=self.data_flag,
                                               data_transform=eval_transforms,
                                               size=self.img_size,
                                               mmap_mode=self.mmap_mode)
        elif stage == 'validate':
            eval_transforms = self._get_transforms(train_transforms=["ToImage", "DoDtype", "Normalize"])[1]
            self.val_ds = get_medmnist_dataset(mode="val",
                                               data_flag=self.data_flag,
                                               data_transform=eval_transforms,
                                               size=self.img_size,
                                               mmap_mode=self.mmap_mode)
        elif stage == 'test':
            eval_transforms = self._get_transforms(train_transforms=["ToImage", "DoDtype", "Normalize"])[1]
            self.test_ds = get_medmnist_dataset(mode="test",
                                                data_flag=self.data_flag,
                                                data_transform=eval_transforms,
                                                size=self.img_size,
                                                mmap_mode=self.mmap_mode)

    def _get_transforms(self, train_transforms: List[str]):
        transforms_dict = {"ToImage": v2.ToImage(),
                           "DoDtype": v2.ToDtype(torch.float32, scale=True),
                           "Normalize": v2.Normalize(mean=self.mean_std[0], std=self.mean_std[1]),
                           "RandomAdjustSharpness": v2.RandomAdjustSharpness(sharpness_factor=choice(linspace(0.5, 1.5, 10)), p=0.5),
                           "ColorJitter": v2.ColorJitter(brightness=.2, hue=.2, contrast=.2, saturation=.2),
                           "RandomVerticalFlip": v2.RandomVerticalFlip(p=0.5),
                           "RandomHorizontalFlip": v2.RandomHorizontalFlip(p=0.5),
                           "RandomResizedCrop": v2.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0)),
                           "GaussianBlur": v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                           "RandomAffine": v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                           "ElasticTransform": T.ElasticTransform(alpha=50.0, sigma=5.0),
                           "Resize": v2.Resize(size=[224, 224])
                           }

        train_transforms = v2.Compose([transforms_dict[key] for key in train_transforms])
        eval_transforms = v2.Compose([transforms_dict[key] for key in ["ToImage", "DoDtype", "Normalize"]])

        return (train_transforms, eval_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
