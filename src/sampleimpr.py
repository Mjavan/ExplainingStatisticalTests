import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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

from models import SimCLR
from embeddingtest import MMDTest

b_size = 16


class SampleImportance:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._read_config()
        self.root_dir = Path(self.base_path)
        self.embed_dir = self.root_dir / 'embeddings'
        self.param_dir = self.root_dir / 'params'
        self.inf_dir = self.root_dir / 'infscores'
        self.param = self._read_param(self.param_dir)

        self._set_random_seed()
        self._load_images()
     
    def _read_config(self,file_path='config.json'):
        with open(file_path, 'r') as file:
            config = json.load(file) 
        for key, value in config.items():
            setattr(self, key ,value)
        return config
    
    def _read_param(self, file_path=None):
        param_path = os.path.join(file_path, str(self.args.exp)+'_params.json')
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"The parameter file '{param_path}' does not exist.")
        with open(param_path, "r") as f:
            params = json.load(f)
        for key, value in params.items():
            setattr(self, key ,value)
        return params
    
    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        self.seed = self.config.get('random_state', 42)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _load_checkpoint(self):
        """Load checkpoints of pretrained model"""
        pre_exp = self.param['pre_exp']
        self.model = self.param['model']
        checkpoint_dir = self.root_dir / 'self_supervised' / 'simclr' / 'simclr_ckpts'
        if self.model == 'simclr':
            sam_dir_last = os.path.join(checkpoint_dir,f'{pre_exp}_last_sclr.pt')
        elif self.model == 'bsimclr':
            sam_dir_last = os.path.join(checkpoint_dir,f'{pre_exp}_last_cbsclr.pt')
        state_dict = torch.load(sam_dir_last,map_location=self.device) 
        print('for model %s, epoch %d'%(self.model,state_dict['epoch']))
        return state_dict
        
    def _load_model(self):
        """Load pre-trained model."""
        self.model = self.param['model']
        try:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            if self.model=='imgnet':
                self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
            elif self.model in ['simclr','bsimclr']:
                model = SimCLR(backbone,hid_dim=2048,out_dim=128).to(self.device)
                state_dict = self._load_checkpoint()
                model.load_state_dict(state_dict['model'])
                print(f'simclr pretrained was loaded.')
                self.encoder = torch.nn.Sequential(*list(model.children())[:-1])            
        except Exception as e:
            print(f"Error loading model: {e}")
        self.encoder.eval()
        return self.encoder
    
    def _load_images(self):
        self.healthy_gr, self.unhealthy_gr = get_groups(self.config)
        self.healthy_loader = DataLoader(self.healthy_gr, batch_size=b_size, shuffle=False, drop_last=True)
        self.unhealthy_loader = DataLoader(self.unhealthy_gr, batch_size=b_size, shuffle=False, drop_last=True)
    
    def _get_embeddings(self, dataloader):
        """Takes dataloader of each group, extract embedding vectors, return embeddings"""
        self._load_model()
        embeddings_list = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                embeddings = self.encoder.forward(images)
                embeddings = embeddings.view(embeddings.size()[0],-1)
                embeddings_list.append(embeddings.cpu().numpy())
        return(np.vstack(embeddings_list))
    
    def _exclude_sample(self,i, dataset):
        """Exclude i'th sample from dataset, return new dataset"""
        # Specify the sample index to exclude
        exclude_idx = i
        # Create a list of all indices except the one to exclude
        all_indices = list(range(len(dataset)))
        remaining_indices = [idx for idx in all_indices if idx != exclude_idx]
        # Create a Subset of the dataset excluding the sample
        updated_dataset = Subset(dataset, remaining_indices)
        # Create a new DataLoader with the updated dataset
        updated_loader = DataLoader(updated_dataset, batch_size=b_size, shuffle=False, drop_last=True)
        return updated_loader
    
    def _get_orig_test_statistic(self):
        embed_X, embed_Y = self.load_embeddings()
        statistic = self._calculate_test_statistics(embed_X,embed_Y)
        print(f'test-statistic for whole dataset is:{statistic:0.4f}')
        return statistic
    
    def _calculate_test_statistics(self, embed_X, embed_Y):
        """Calculate the test statistic for different groups of DR."""
        # Calculate mean embeddings
        mean_X = embed_X.mean(0)
        mean_Y = embed_Y.mean(0)
        D = mean_X - mean_Y
        statistic = np.linalg.norm(D)**2
        return statistic

    def load_embeddings(self):
        """Take embeddings of original data set"""
        try:
            # Get the sample sizes
            n = self.sample_size.get('healthy_size', 100)
            m = self.sample_size.get('unhealthy_size', 100)

            # Build file paths
            file_path = os.path.join(self.embed_dir, f'{self.seed}_{self.pre_exp}_{self.model}_{n+m}')
            os.makedirs(file_path, exist_ok=True)

            path_X = os.path.join(file_path, 'healthy_embed.npy')
            path_Y = os.path.join(file_path, 'unhealthy_embed.npy')

            # Check if embedding files exist
            if os.path.exists(path_X) and os.path.exists(path_Y):
                print("Loading embeddings from saved numpy files...")
                try:
                    # Load numpy arrays from disk with memmap
                    embed_X = np.load(path_X, mmap_mode='r')
                    embed_Y = np.load(path_Y, mmap_mode='r')
                    return embed_X, embed_Y
                except Exception as e:
                    raise RuntimeError(f"Error loading .npy files: {e}")
            else:
                raise FileNotFoundError(f"One or both embedding files not found: {path_X}, {path_Y}")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None
    
    def _get_inf_score(self, dataset, D, embed_O):
        inf_list = []
        for i in range(len(dataset)):
            new_dataloader = self._exclude_sample(i, dataset)
            embed_N = self._get_embeddings(new_dataloader)
            d_new = self._calculate_test_statistics(embed_O,embed_N)
            inf_score = D - d_new
            inf_list.append(inf_score)            
        return inf_list
         
    def compute_all_inf_score(self): 
        D = self._get_orig_test_statistic()
        embed_X, embed_Y = self.load_embeddings()        
        inf_list_X = self._get_inf_score(self.healthy_gr, D, embed_Y)
        inf_list_Y = self._get_inf_score(self.unhealthy_gr, D, embed_X)        
        # save influnece_score
        print(f'inf_list_X:{inf_list_X}')
        print(f'inf_list_Y:{inf_list_Y}')
        
        # Build file paths
        n = len(inf_list_X)
        m = len(inf_list_Y)
        file_path = os.path.join(self.inf_dir, f'{self.seed}_{self.pre_exp}_{self.model}_{n+m}')
        os.makedirs(file_path, exist_ok=True)
        
        path_X = os.path.join(file_path,'healthy_inf.npy')
        path_Y = os.path.join(file_path,'unhealthy_inf.npy')
        
        np.save(np.array(inf_list_X),path_X)
        np.save(np.array(inf_list_Y),path_Y)
        
        return inf_list_X,inf_list_Y 
    
    
                                    
parser = argparse.ArgumentParser(description='Computing Sample_Importance')    
parser.add_argument('--exp', type=int, default=7, help='The exp that we used to visualise feature_importnaces!')
args = parser.parse_args()
            
if __name__=="__main__":
    
    test = SampleImportance(args)
    inf_X, inf_Y = test.compute_all_inf_score()



        
        
        
        
        
