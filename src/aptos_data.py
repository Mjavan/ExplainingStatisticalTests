import os
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
from scipy import ndimage
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import glob
from joblib import Parallel, delayed
from typing import Tuple, Optional


import cv2
from skimage.draw import circle_perimeter

with open('config.json', 'r') as config_file:
    config = json.load(config_file)


class APTOS(Dataset):
    """
    Dataset for aptos for pretraining in self-supervised manner.

    Parameters:
    base_path (str): The base path to the dataset directory.
    train_size (float): The percentage of data that we want to take for training.
    split (str): Which split of data we want to make
    tfs (callable, optional): Transformations to be applied to the images.
    split_seed: seed for reproducibility.
    """

    def __init__(self, base_path, train_size, split, tfs, split_seed):
        self.train_path = os.path.join(base_path, 'Aptos', 'cropped_train')
        self.total_list = os.listdir(self.train_path)
        self.tfs = tfs
        rng = np.random.RandomState(split_seed)
        N = len(self.total_list)
        perm = rng.permutation(N)
        m = int(N * train_size)
        if split == 'train':
            self.img_list = [self.total_list[i] for i in perm[:m]]
        elif split == 'val':
            self.img_list = [self.total_list[i] for i in perm[m:]]
        else:
            raise ValueError("Invalid split: choose 'train' or 'val'")

    def __len__(self) -> int:
        return len(self.img_list)

    def _load_item(self, idx: int) -> Image.Image:
        img_idx = self.img_list[idx]
        img_path = os.path.join(self.train_path, img_idx)
        try:
            return Image.open(img_path)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {img_idx}: {e}")
            return None

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        img = self._load_item(idx)
        if img is None:
            raise RuntimeError(f"Image at index {idx} could not be loaded.")
        if self.tfs:
            img1 = self.tfs(img)
            img2 = self.tfs(img)
        return img1, img2


def get_tfs(size=224, setting="none"):
    """Getting transformations for pretraining"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if setting == "none":
        tfms = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif setting == "simclr":
        cj_prob = 0.8
        cj_bright = 0.7
        cj_contrast = 0.7
        cj_sat = 0.7
        cj_hue = 0.2
        min_scale = 0.08
        random_gray_scale = 0.2
        gaussian_blur = 0.5
        kernel_size = 0.1
        hf_prob = 0.5
        color_jitter = transforms.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue)
        tfms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=(min_scale, 1.0)),
                transforms.RandomHorizontalFlip(p=hf_prob),
                transforms.RandomApply([color_jitter], p=cj_prob),
                transforms.RandomGrayscale(p=random_gray_scale),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    elif setting == 'contig':

        tfms = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return tfms


def get_aptos_loader(
        base_path,
        size,
        split_seed=42,
        batch_size=64,
        num_workers=6,
        train_size=0.8,
        val_size=0.2,
        aug='none'):

    assert 0 < train_size <= 1, "train_size must be in range (0, 1]"
    assert train_size + val_size <= 1, "train_size + val_size must be <= 1"

    loaders = []
    if val_size:
        for split in ['train', 'val']:
            if split == 'train':
                tfs_dl = get_tfs(size=size, setting=aug)
            else:
                tfs_dl = get_tfs(size=size, setting='none')
            D = APTOS(
                base_path,
                train_size,
                split=split,
                tfs=tfs_dl,
                split_seed=split_seed)

            loader = DataLoader(
                D,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=num_workers,
                pin_memory=True,
            )
            loaders.append(loader)
    else:
        tfms = get_tfs(size=size, setting=aug)
        D = APTOS(
            base_path,
            train_size,
            split='train',
            tfs=tfms,
            split_seed=split_seed)

        loader = DataLoader(
            D,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders


def crop_to_circle(image_path):
    """Detect circles to eliminate redundant corners"""
    image = cv2.imread(image_path)
    if image is None:
        flag = False
        return Image.open(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=100,
        maxRadius=gray.shape[0] // 2
    )
    if circles is None:
        flag = False
        return Image.open(image_path)

    circles = np.round(circles[0, :]).astype("int")
    x, y, r = circles[0]
    h, w = gray.shape
    if y-r < 0 or y+r > h or x-r < 0 or x+r > w:
        flag = False
        return Image.open(image_path)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), r, 255, thickness=-1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    cropped_image = masked_image[y-r:y+r, x-r:x+r]
    if cropped_image.size == 0:
        flag = False
        return Image.open(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    return pil_image


def process_image(file_path, save_path):
    """
    Process a single image: Crop it and save it to the output folder.
    """
    try:
        img_path = file_path
        if img_path.lower().endswith(('.png')):
            cropped_img, proc = crop_to_circle(img_path)
            if not proc:
                save_file_path = os.path.join(
                    save_path, f"failed_{os.path.basename(file_path)}")
            else:
                save_file_path = os.path.join(
                    save_path, f"cropped_{os.path.basename(file_path)}")
            cropped_img.save(save_file_path)
            return None
    except Exception as e:
        return file_path


def crop_all_images(path, save_path, num_workers=10):
    """
    Crop all images and save to a save_path directory.
    """
    path_list = [os.path.join(path, fname) for fname in os.listdir(path)]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Parallel(n_jobs=num_workers)(delayed(process_image)(
        img_path, save_path) for img_path in path_list)
