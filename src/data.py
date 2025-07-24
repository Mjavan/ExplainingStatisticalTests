import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import os
from PIL import Image

import json
from pathlib import Path
from typing import Tuple, Optional

with open('config.json', 'r') as config_file:
        config = json.load(config_file)



class DiabeticRetinopathy(Dataset):
    """
    Dataset for diabetic retinopathy images and their labels.

    Parameters:
    base_path (str): The base path to the dataset directory.
    tfs (callable, optional): Transformations to be applied to the images.
    """
    def __init__(self, base_path: str, tfs: Optional[callable] = None):
        self.train_path = os.path.join(base_path,'Retina','train')
        self.labels = pd.read_csv(os.path.join(base_path,'Retina','trainLabels.csv'))
        self.tfs = tfs
        
    def __len__(self) -> int:
        return len(self.labels)
        
    def _load_item(self, idx: int) -> Tuple[Image.Image, int]:
        img_idx = self.labels.iloc[idx]['image']
        label = self.labels.iloc[idx]['level']
        try:
            img = Image.open(os.path.join(self.train_path, img_idx + '.jpeg'))
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_idx}: {e}")
            return None, label
        return img, label
             
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img, label = self._load_item(idx)
        if self.tfs:
            img = self.tfs(img)
        return img, label
    
def stratified_sampledataset(dataset: DiabeticRetinopathy, healthy_size: int, unhealthy_size: int, rnd_st: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Separate healthy samples (class 0)
    healthy_samples = dataset.labels[dataset.labels['level'] == 0]
    healthy_sampled = healthy_samples.sample(n=healthy_size, random_state=rnd_st).reset_index(drop=True)
    
    # Separate unhealthy samples (classes 1-4)
    unhealthy_samples = dataset.labels[dataset.labels['level'].isin([1, 2, 3, 4])]
    
    # Determine total counts for stratified sampling
    total_unhealthy = unhealthy_samples['level'].value_counts()
    total_samples = unhealthy_size
    
    # Calculate the proportion of samples for each class => A series
    proportions = (total_unhealthy / total_unhealthy.sum()) * total_samples
    proportions = proportions.round().astype(int)  # Round to get integer counts
    
    unhealthy_sampled = []
    for label, count in proportions.items():
        if count > 0:  # Ensure we only sample if there are available samples
            class_samples = unhealthy_samples[unhealthy_samples['level'] == label]
            # it samples n rows from the rows of dataframe
            sampled_class = class_samples.sample(n=count, random_state=rnd_st)
            unhealthy_sampled.append(sampled_class)
    
    unhealthy_sampled_df = pd.concat(unhealthy_sampled).reset_index(drop=True)
    return healthy_sampled, unhealthy_sampled_df 
        

class SampledDiabeticRetinopathy(Dataset):
    def __init__(self, dataframe: pd.DataFrame, base_path: str,
                  tfs: Optional[callable] = None):
        self.dataframe = dataframe.reset_index(drop=True)        
        self.base_path = base_path
        self.tfs = tfs

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_idx = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['level']
        img = Image.open(os.path.join(self.base_path,'Retina','train', img_idx + '.jpeg'))
        if self.tfs:
            img = self.tfs(img)
        return img, label
                
        
def get_tfms(size: int = 224, interpolation = Image.BILINEAR) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=mean, std=std)
    trsf = transforms.Compose([
        transforms.Resize(size=(size,size)),
        transforms.ToTensor(),
        norm,
        ])
    return trsf


def get_groups(config: dict = config) -> Tuple[SampledDiabeticRetinopathy, SampledDiabeticRetinopathy]:
    """
    Groups from diabetic retinopathy healthy(class=0) and unhealthy(class=1..4).

    Parameters:
    config (dict): A dictionary containing the parameters.
    """
    
    base_path = Path(config['base_path'])
    healthy_size = config['sample_size']['healthy_size']
    unhealthy_size = config['sample_size']['unhealthy_size']
    random_state = config['random_state']
    size = config['size']
    
    full_dataset = DiabeticRetinopathy(base_path=base_path, tfs=get_tfms(size=size))
    healthy, unhealthy = stratified_sampledataset(full_dataset, healthy_size, unhealthy_size, rnd_st=random_state)
    
    healthy_ds = SampledDiabeticRetinopathy(healthy, base_path=base_path, tfs=get_tfms())
    unhealthy_ds = SampledDiabeticRetinopathy(unhealthy, base_path=base_path, tfs=get_tfms())
    
    return(healthy_ds,unhealthy_ds)
    
    
    
