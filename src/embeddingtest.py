from scipy import stats
import numpy as np
import torch
import argparse
import json
from pathlib import Path
import os
import random
from data import get_groups
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

class MMDTest:
    def __init__(self, features_X, features_Y, n_perm=1000):
        
        self.n_perm = n_perm
        self.features_X = features_X
        self.features_Y = features_Y
        
    def _compute_mmd(self, features_X, features_Y):
        
        mean_fX = features_X.mean(0)
        mean_fY = features_Y.mean(0)
        D = mean_fX - mean_fY
        statistic = np.linalg.norm(D)**2
        return statistic
    
    def _compute_p_value(self):
        
        # compute real test statistic
        stat = self._compute_mmd(self.features_X, self.features_Y)
        n, m = len(self.features_X), len(self.features_Y)
        l = n + m
        features_Z = np.vstack((self.features_X, self.features_Y))
        
        # compute null samples
        resampled_vals = np.empty(self.n_perm)
        for i in range(self.n_perm):
            index = np.random.permutation(l) # it permutes indices from 0 to l
            feats_X, feats_Y = features_Z[index[:n]], features_Z[index[n:]]
            resampled_vals[i] = self._compute_mmd(feats_X, feats_Y)
            
        resampled_vals.sort()
        #p_val = np.mean(stat < resampled_vals)
        p_val = (np.sum(stat<= resampled_vals)+1)/(self.n_perm+1)
        return p_val
    
    def test(self):
        return(self._compute_p_value())
    
class Test:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._read_config()
        self.root_dir = Path(self.base_path)
        self.embed_dir = self.root_dir / 'embeddings'
        self.param_dir = self.root_dir / 'params'
        self._read_param(self.param_dir)
        self._set_random_seed()
     
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
        
    def _load_model(self):
        """Load pre-trained model."""
        try:
            if self.model=='resnet50-imgnet':
                print(f'model was loaded')
                # Attempt to load a pretrained model
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
        self.encoder.eval()
        return self.encoder
    
    def _load_images(self):

        healthy_gr, unhealthy_gr = get_groups(self.config)
        healthy_loader = DataLoader(healthy_gr, batch_size=self.bs, shuffle=False, drop_last=True)
        unhealthy_loader = DataLoader(unhealthy_gr, batch_size=self.bs, shuffle=False, drop_last=True)
        return healthy_loader, unhealthy_loader 
        
    def _get_embeddings(self, model, dataloader):
        """Takes dataloader of each group, extract embedding vectors, return embeddings"""
        # Initialize accumulators for healthy and unhealthy groups
        embeddings_list = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                embeddings = model.forward(images)
                embeddings = embeddings.view(embeddings.size()[0],-1)
                embeddings_list.append(embeddings.cpu().numpy())
        return(np.vstack(embeddings_list))
    
    def load_save_embeddings(self):
        """ we make directory and save embeddinsg"""
        n,m = self.sample_size.get('healthy_size', 100), self.sample_size.get('unhealthy_size', 100)
        file_path = os.path.join(self.embed_dir, f'{self.seed}_{self.pre_exp}_{self.model}_{n+m}')
        os.makedirs(file_path, exist_ok=True)
        
        path_X = os.path.join(file_path,'healthy_embed.npy')
        path_Y = os.path.join(file_path ,'unhealthy_embed.npy')

        # check if we did not save mebdiings before
        if os.path.exists(path_X) and os.path.exists(path_Y):
            print("Loading embeddings from saved numpy files...")
            # Load numpy arrays from disk with memmap
            embed_X = np.load(path_X, mmap_mode='r')
            embed_Y = np.load(path_Y, mmap_mode='r')
        
        else:
            print("Saving extrected embeddings as numpy arrays if they were not save before")
            model = self._load_model()
            dataloader_X, dataloader_Y = self._load_images() 
            embed_X = self._get_embeddings(model,dataloader_X)
            embed_Y = self._get_embeddings(model,dataloader_Y)
            np.save(path_X, embed_X)
            np.save(path_Y, embed_Y)
        
        return embed_X, embed_Y

parser = argparse.ArgumentParser(description='Computing p-value')    
parser.add_argument('--exp', type=int, default=7)
args = parser.parse_args()
            
if __name__=="__main__":
    
    test = Test(args)
    embed_X, embed_Y = test.load_save_embeddings()
    mmd_test = MMDTest(embed_X, embed_Y)
    pvalue = mmd_test.test()
    
    print(f'pvalue is:{pvalue: 0.5f}')
    
    
    
            
            
        
        
        
        