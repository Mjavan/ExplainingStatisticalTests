import torch
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import argparse
import random
from typing import Tuple

from data import get_groups
from utils import overlay_heatmap_single


class HeatmapOverlay:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.config = self._read_config()
        self._initialize_directories()
        self._set_random_seed()
        # Initialize attributes for images and heatmaps
        self.healthy_np, self.unhealthy_np = None, None
        self.h_X, self.h_Y = None, None

        self._load_images()
        self._load_heatmaps()

    def _read_config(self, file_path='config.json'):
        with open(file_path, 'r') as file:
            config = json.load(file)
        for key, value in config.items():
            setattr(self, key, value)
        return config

    def _initialize_directories(self):
        """Set up root and subdirectories for outputs and experiments."""
        self.root_dir = Path(self.config['base_path'])
        self.heatmap_dir = self.root_dir / 'heatmaps'
        self.param_dir = self.root_dir / 'params'
        self.overlay_dir = self.root_dir / 'overlays'
        self.img_dir = self.root_dir / 'images'

        # Load experiment parameters
        self.params = self._read_param(self.param_dir)
        for key, value in self.params.items():
            setattr(self, key, value)

    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        self.seed = self.config.get('random_state', 42)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _read_param(self, file_path=None):
        param_path = os.path.join(file_path, str(self.args.exp)+'_params.json')
        # Check if the file exists before trying to read it
        if not os.path.exists(param_path):
            raise FileNotFoundError(
                f"The parameter file '{param_path}' does not exist.")
        with open(param_path, "r") as f:
            params = json.load(f)
        return params

    def _load_heatmaps(self):
        """Load precomputed heatmaps with memory-mapping for efficiency."""
        base_path = os.path.join(
            self.heatmap_dir, f'{self.seed}_{self.args.exp}_{self.model}_{self.target_layer}')
        try:
            self.h_X = np.load(os.path.join(
                base_path, f'healthy_{self.relu}_heatmap.npy'), mmap_mode='r')
            self.h_Y = np.load(os.path.join(
                base_path, f'unhealthy_{self.relu}_heatmap.npy'), mmap_mode='r')
        except FileNotFoundError as e:
            print(f"Error loading heatmaps: {e}")

    def _get_file_paths(self, file_name: str, prefix: str):
        """Generate paths for image and label numpy files based on a prefix."""
        file_path = os.path.join(file_name, f'{prefix}_images.npy')
        label_path = os.path.join(file_name, f'{prefix}_labels.npy')
        return file_path, label_path

    def _dataloader_to_numpy(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        images, labels = [], []
        for img_batch, label_batch in dataloader:
            # Convert torch tensors to numpy arrays
            images.append(img_batch.cpu().numpy())
            labels.append(label_batch.cpu().numpy())
        # Shape: (num_samples, height, width, channels)
        images_np = np.concatenate(images)
        labels_np = np.concatenate(labels)  # Shape: (num_samples,)
        return images_np, labels_np

    def _load_images(self):
        '''Load images in test set and save them'''
        # Define directories to save images as numpy arrays
        h_size = self.sample_size.get('healthy_size', 100)
        unh_size = self.sample_size.get('unhealthy_size', 100)
        file_name = os.path.join(
            self.img_dir, f'{self.seed}_{h_size}_{unh_size}')
        os.makedirs(file_name, exist_ok=True)

        file_healthy, label_healthy = self._get_file_paths(
            file_name, 'healthy')
        file_unhealthy, label_unhealthy = self._get_file_paths(
            file_name, 'unhealthy')

        if os.path.exists(file_healthy) and os.path.exists(file_unhealthy):
            print("Loading images from saved numpy files...")
            # Load numpy arrays from disk with memmap
            self.healthy_np = np.load(file_healthy, mmap_mode='r')
            self.unhealthy_np = np.load(file_unhealthy, mmap_mode='r')
        else:
            print("Loading images from dataloaders and saving as numpy files...")
            # Load data from dataloaders
            healthy_gr, unhealthy_gr = get_groups(self.config)
            healthy_loader = DataLoader(
                healthy_gr, batch_size=self.bs, shuffle=False, drop_last=True)
            unhealthy_loader = DataLoader(
                unhealthy_gr, batch_size=self.bs, shuffle=False, drop_last=True)
            self.healthy_np, healthy_labels_np = self._dataloader_to_numpy(
                healthy_loader)
            self.unhealthy_np, unhealthy_labels_np = self._dataloader_to_numpy(
                unhealthy_loader)
            np.save(file_healthy, self.healthy_np)
            np.save(file_unhealthy, self.unhealthy_np)
            np.save(label_healthy, healthy_labels_np)
            np.save(label_unhealthy, unhealthy_labels_np)
            print("Images and labels saved as numpy files.")

    def prepare_overlay_directory(self):
        """Creates the overlay directory to save overlaid images."""
        base_path = os.path.join(
            self.overlay_dir, f'{self.seed}_{self.exp}_{self.model}_{self.target_layer}')
        os.makedirs(base_path, exist_ok=True)

        file_name1 = f'healthy_{self.relu}_overlays'
        file_name2 = file_name1.replace('healthy', 'unhealthy')

        healthy_path_ov = os.path.join(base_path, file_name1)
        unhealthy_path_ov = os.path.join(base_path, file_name2)

        os.makedirs(healthy_path_ov, exist_ok=True)
        os.makedirs(unhealthy_path_ov, exist_ok=True)
        return healthy_path_ov, unhealthy_path_ov

    def visualise_overlay(self):
        """Visualize and save overlay of heatmap on images in batch mode."""
        # Select the group (healthy/unhealthy) based on the argument
        image_group = self.healthy_np if self.args.gr == 'X' else self.unhealthy_np
        heatmap_group = self.h_X if self.args.gr == 'X' else self.h_Y
        # Prepare directories to save overlays
        self.health_dir, self.unhealth_dir = self.prepare_overlay_directory()
        # Determine the range of indices to process in the batch
        start_idx = self.args.img_idx
        # Ensure we don't go out of bounds
        end_idx = min(start_idx + self.args.batch_size, len(image_group))

        # Loop over the batch of images
        for idx in range(start_idx, end_idx):
            image = image_group[idx]
            heatmap = heatmap_group[idx]
            # Determine the save directory based on the group
            save_dir = self.health_dir if self.args.gr == 'X' else self.unhealth_dir
            # Overlay the heatmap on the image and save it
            overlay_heatmap_single(image, heatmap, idx,
                                   save_dir, alpha=self.args.alpha)


parser = argparse.ArgumentParser(description='Overlaying HeatMaps')
parser.add_argument('--exp', type=int, default=11,
                    help='ID of the current experiment!')
parser.add_argument('--gr', type=str, default='X',
                    help='The group that we want to visualise')
parser.add_argument('--img_idx', type=int, default=0,
                    help='The index of image that we want to overlay heatmap on!')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for processing images')
parser.add_argument('--alpha', type=float, default=0.4,
                    help='The factor used to blend the Grad-CAM heatmap with the original image')
args = parser.parse_args()

if __name__ == '__main__':

    heatmap_overlay = HeatmapOverlay(args)
    heatmap_overlay.visualise_overlay()
