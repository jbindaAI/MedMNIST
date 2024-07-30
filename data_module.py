import pytorch_lightning as pl
from medmnist_dataset import get_medmnist_dataset, DataFlag
from typing import List, Literal, Optional, Tuple
from torch.utils.data import DataLoader
from monai import transforms as T
from torchvision.transforms import v2
from random import choice
from numpy import linspace
import torch
import pickle
import os 


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_flag:DataFlag,
                 img_size:Literal[28,64,128,224]=28,
                 batch_size:int=32,
                 num_workers:int=8,
                 selected_train_transforms:List[str]=["ToImage", "DoDtype","Normalize", "RandomResizedCrop"],
                 mmap_mode:Optional[Literal["r"]]=None
                ):
        super().__init__()
        self.data_flag=data_flag
        self.batch_size=batch_size
        self.img_size=img_size
        self.num_workers=num_workers
        self.selected_train_transforms=selected_train_transforms
        self.mmap_mode=mmap_mode

    
    def prepare_data(self):
        # Downloading data
        temp_ds = get_medmnist_dataset(data_flag=self.data_flag,
                                       size=self.img_size,
                                       download=True)

    
    def _get_mean_std(self, batch_size=32)->Tuple[torch.Tensor, torch.Tensor]:
        dataset = get_medmnist_dataset(mode="train",
                                       data_flag=self.data_flag,
                                       size=self.img_size,
                                       data_transform=v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)]),
                                       download=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Computing mean and standard deviation:
        channel_sum = torch.zeros(3)
        channel_sum_of_squares = torch.zeros(3)
        pixel_count = 0
        for images, _ in loader:
            # Accumulate the sum of pixel values for each channel
            channel_sum += torch.sum(images, dim=(0, 2, 3))
            # Accumulate the sum of squared pixel values for each channel
            channel_sum_of_squares += torch.sum(images ** 2, dim=[0, 2, 3])
            # Accumulate the count of images
            pixel_count += images.size(0)*images.size(2)*images.size(3)

        # Compute the mean by dividing the sum of pixel values by the pixel count
        channel_mean = channel_sum / pixel_count
        # Compute the standard deviation using the sum of squares
        channel_std = torch.sqrt((channel_sum_of_squares / pixel_count) - (channel_mean ** 2))
        
        with open(f"fitted_factors/{self.data_flag.value}.pkl", "wb") as file:
            pickle.dump((channel_mean, channel_std), file)
            
        return (channel_mean, channel_std)


    def _get_transforms(self, train_transforms:List[str]):
        transforms_dict = {"ToImage": v2.ToImage(),
                           "DoDtype": v2.ToDtype(torch.float32, scale=True),
                           "Normalize": v2.Normalize(mean=self.mean_std[0], std=self.mean_std[1]),
                           "RandomAdjustSharpness": v2.RandomAdjustSharpness(sharpness_factor=choice(linspace(0.5, 1.5, 10)), p=0.5),
                           #"GaussianNoise": v2.GaussianNoise(mean=0, sigma=0.1),
                           "ColorJitter": v2.ColorJitter(brightness=.2, hue=.2, contrast=.2, saturation=.2),
                           "RandomVerticalFlip": v2.RandomVerticalFlip(p=0.5),
                           "RandomHorizontalFlip": v2.RandomHorizontalFlip(p=0.5),
                           "RandomResizedCrop": v2.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0))
                          }
        
        train_transforms=v2.Compose([transforms_dict[key] for key in train_transforms])
        eval_transforms=v2.Compose([transforms_dict[key] for key in ["ToImage", "DoDtype", "Normalize"]])
        
        return (train_transforms, eval_transforms)
      
    
    def setup(self, stage=None):

        try:
            with open(f"fitted_factors/{self.data_flag.value}.pkl", "rb") as file:
                self.mean_std = pickle.load(file)
        except FileNotFoundError:
            print("Calculating mean and standard deviation.")
            self.mean_std = self._get_mean_std()
        
        train_transforms, eval_transforms = self._get_transforms(train_transforms=self.selected_train_transforms)
        
        self.train_ds = get_medmnist_dataset(mode="train", data_flag=self.data_flag, data_transform=train_transforms, size=self.img_size)
        self.val_ds = get_medmnist_dataset(mode="val", data_flag=self.data_flag, data_transform=eval_transforms, size=self.img_size)
        self.test_ds = get_medmnist_dataset(mode="test", data_flag=self.data_flag, data_transform=eval_transforms, size=self.img_size)


    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers
                                 )
        return train_loader

    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds,
                                shuffle=False,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers
                               )
        return val_loader

    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds,
                                shuffle=False,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers
                               )
        return test_loader
