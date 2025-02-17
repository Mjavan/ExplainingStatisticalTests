
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import argparse
    
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from data import get_groups
from gradcam import GradCAM
from models import SimCLR

    
class TestStatisticBackprop:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = self._read_config()
        print(self.config)
        self._setup_experiment()
        
    def _read_config(self,file_path='config.json'):
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config

    def _setup_experiment(self):
        """Set random seeds, directories, and test loader."""
        for key, value in self.config.items():
            setattr(self, key, value)
        self.seed = self.random_state
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Directory for loading checkpoints and saving outputs
        self.root_dir = Path(self.base_path) 
        self.heatmap_dir = os.path.join(self.root_dir, 'heatmaps') 
        self.embed_dir = os.path.join(self.root_dir, 'embeddings')
        self.param_dir = os.path.join(self.root_dir, 'params')
        self._save_param()
        self._load_test_set()
        
    def _save_param(self):
        args_dict = vars(self.args)
        param_path = os.path.join(self.param_dir, f'{self.args.exp}')
        with open(f"{param_path}_params.json","w") as f:
            json.dump(args_dict, f, indent=4)                

    def _load_test_set(self):
        """Load test dataset."""
        healthy_gr, unhealthy_gr = get_groups(self.config)
        self.healthy_loader = DataLoader(healthy_gr, batch_size=self.args.bs, shuffle=False, drop_last=True)
        self.unhealthy_loader = DataLoader(unhealthy_gr, batch_size=self.args.bs, shuffle=False, drop_last=True)
        print(f'len healthy: {len(self.healthy_loader.dataset)}, num_batches:{len(self.healthy_loader)}')
        print(f'len unhealthy: {len(self.unhealthy_loader.dataset)}, num_batches:{len(self.unhealthy_loader)}')
        
    def _load_checkpoint(self):
        """Load checkpoints of pretrained model"""
        checkpoint_dir = self.root_dir / 'self_supervised' / 'simclr' / 'simclr_ckpts'
        if self.args.model == 'simclr':
            sam_dir_last = os.path.join(checkpoint_dir,f'{self.args.pre_exp}_last_sclr.pt')
        elif self.args.model == 'bsimclr':
            sam_dir_last = os.path.join(checkpoint_dir,f'{self.args.pre_exp}_last_cbsclr.pt')
        state_dict = torch.load(sam_dir_last,map_location=self.device) 
        print('for model %s, epoch %d'%(self.args.model,state_dict['epoch']))
        return state_dict
        
    def _load_model(self):
        """Load pre-trained model."""
        try:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            if self.args.model=='imgnet':
                # Attempt to load a pretrained model with imagenet
                self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
            elif self.args.model in ['simclr','bsimclr']:
                model = SimCLR(backbone,hid_dim=2048,out_dim=128).to(self.device)
                state_dict = self._load_checkpoint()
                model.load_state_dict(state_dict['model'])
                print(f'simclr pretrained was loaded.')
                self.encoder = torch.nn.Sequential(*list(model.children())[:-1])            
        except Exception as e:
            print(f"Error loading model: {e}")
        self.encoder.eval()
        return self.encoder
    
    def _get_mean_embeddings(self,gcam, dataloader):
        """Takes dataloader of each group, extract embedding vectors, return mean embeddings"""
        # Initialize accumulators for healthy and unhealthy groups
        sum_f = torch.zeros_like(torch.zeros(self.args.latent_dim)).to(self.device)  
        count_f = 0 # Count of samples in each group
        for images, _ in dataloader:
            images = images.to(self.device)
            embeddings = gcam.forward(images)
            embeddings = embeddings.view(embeddings.size()[0],-1)
            sum_f += embeddings.sum(dim=0)  # Sum of embeddings for this batch
            count_f += embeddings.size(0) 
        mean_embed = sum_f / count_f if count_f > 0 else torch.zeros_like(sum_f) 
        del sum_f
        torch.cuda.empty_cache()
        return mean_embed
                    
    def calculate_test_statistics(self, model):
        """Calculate the test statistic for different groups of DR."""
        if self.args.model in['simclr','bsimclr']:
            target_layer = '0.'+self.args.target_layer
        else:
            target_layer = self.args.target_layer    
        # Instantiate GradCAM for feature attribution
        gcam = GradCAM(model, target_layer=target_layer, relu=self.args.relu, device=self.device)
        # Calculate mean embeddings
        healthy_mean = self._get_mean_embeddings(gcam, self.healthy_loader)
        unhealthy_mean = self._get_mean_embeddings(gcam, self.unhealthy_loader)
        D = healthy_mean - unhealthy_mean
        print(f'healthy_mean:{healthy_mean.shape}')
        print(f'unhealthy_mean:{unhealthy_mean.shape}')
        test_statistic = torch.norm(D, p=2)
        return(test_statistic, D, gcam)
    
    def process_attributions(self, dataloader, gcam, backprop_value):
        """Process and return GradCAM attributions in batches."""
        attributions_list = []
        embed_list = [] # save embeddings as numpy array
        # Compute attribution maps for each group group
        for images, _ in dataloader:
            images = images.to(self.device)
            embeddings = gcam.forward(images)
            embeddings = embeddings.view(embeddings.size()[0],-1).cpu().data.numpy()
            embed_list.append(embeddings)
            del embeddings
            gcam.backward(backprop_value)
            attributions = gcam.generate()
            attributions = attributions.squeeze().cpu().data.numpy()
            attributions_list.append(attributions)
        return np.vstack(attributions_list), np.vstack(embed_list) 
    
    def save_embeddings(self, embed_X, embed_Y):
        n,m = self.sample_size.get('healthy_size', 100), self.sample_size.get('unhealthy_size', 100)
        file_path = os.path.join(self.embed_dir, f'{self.seed}_{self.args.pre_exp}_{self.args.model}_{n+m}')
        os.makedirs(file_path, exist_ok=True)
        
        path_X = os.path.join(file_path,'healthy_embed.npy')
        path_Y = os.path.join(file_path ,'unhealthy_embed.npy')
        
        if not os.path.exists(path_X) or not os.path.exists(path_Y):
            np.save(path_X, embed_X)
            np.save(path_Y, embed_Y)
            print('embeddings were saved')
            
        
    def save_attributions(self, att_g1, att_g2, latent_dim_idx=None):
        """Save the generated attributions to file."""
        base_path = os.path.join(self.heatmap_dir, f'{self.seed}_{self.args.exp}_{self.args.model}_{self.args.target_layer}')
        os.makedirs(base_path, exist_ok=True)
        if latent_dim_idx:
            file_name1 = f'healthy_{self.args.relu}_{latent_dim_idx}_heatmap.npy'
            file_name2 = file_name1.replace('healthy', 'unhealthy')
        else:
            file_name1 = f'healthy_{self.args.relu}_heatmap.npy'
            file_name2 = file_name1.replace('healthy', 'unhealthy')
        full_path1 = os.path.join(base_path, file_name1)
        full_path2 = os.path.join(base_path, file_name2)
        np.save(full_path1, att_g1)
        np.save(full_path2, att_g2)
    
    def run(self, backprop_type='test_statistic', latent_dim_idx=None):
        """Main experiment function."""
        model = self._load_model()
        test_statistic, D, gcam = self.calculate_test_statistics(model)
        print(f'test_statistic: {test_statistic:0.4f}\n')

        # Determine which value to backpropagate (test-statistic or specific latent dimension)
        if backprop_type == 'test_statistic':
            backprop_value = test_statistic
        elif backprop_type == 'latent_dim':
            if latent_dim_idx is None or latent_dim_idx >= self.args.latent_dim:
                raise ValueError(f"Invalid latent dimension index: {latent_dim_idx}")
            backprop_value = D[latent_dim_idx]
        else:
            raise ValueError("Invalid backpropagation type. Choose 'test_statistic' or 'latent_dim'.")
        
        attr_healthy, embed_X = self.process_attributions(self.healthy_loader, gcam, backprop_value)
        attr_unhealthy, embed_Y = self.process_attributions(self.unhealthy_loader, gcam, backprop_value) 
        
        self.save_attributions(attr_healthy, attr_unhealthy,latent_dim_idx)
        
        if self.args.save_embed:
            self.save_embeddings(embed_X, embed_Y)
    
    
parser = argparse.ArgumentParser(description='Visualizing Two-Sample Test Retina')
# Model parameters
parser.add_argument('--model', type=str, default='simclr', choices=('simclr','bsimclr','imgnet'))
parser.add_argument('--exp', type=int, default=7)
parser.add_argument('--pre_exp', type=int, default=2, help='The experiment id for pretraining.')
parser.add_argument('--bs', type=int, default=16)  
# Model parameters
parser.add_argument('--latent_dim', type=int, default=2048, help='latent vector size of encoder')
parser.add_argument('--target_layer', type=str, default='7.2.conv3', choices=('7.2.conv1', '7.2.conv2', '7.2.conv3', '7.1.conv1', '7.1.conv2', '7.1.conv3', '7.0.conv1', '7.1.conv1','7.2.conv1')) 
parser.add_argument('--backprop_type', type=str, default='test_statistic', choices= ('test_statistic','latent_dim'))
parser.add_argument('--latent_dim_idx', type=int, default=None)
parser.add_argument('--save_embed', type=bool, default=True)
# Heatmap visualizations
parser.add_argument('--relu', type=bool, default=True, help='If we apply relu on heatmaps in GradCAM!')               
args = parser.parse_args()


def main(args):
    backprop_test = TestStatisticBackprop(args)
    backprop_test.run(backprop_type=args.backprop_type, latent_dim_idx=args.latent_dim_idx)
            
if __name__ == '__main__':
    main(args)



