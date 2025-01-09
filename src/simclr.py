import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

import random
import numpy as np
import os
from pathlib import Path
import json
import time


from models import SimCLR
from aptos_data import get_aptos_loader
from lr_scheduler import CycCosineSch
from utils import plotCurves
from optimizer import SGHM

HID_DIM = 2048
PROJECTION_DIM = 128


##### NT_Xent loss
class NT_Xent(nn.Module):
    def __init__(self, temperature, device):
        super(NT_Xent, self).__init__()        
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
    def forward(self, z_i, z_j): 
        self.batch_size= z_i.size()[0]
        self.mask = torch.ones((self.batch_size * 2, self.batch_size * 2), dtype=bool)
        self.mask = self.mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            self.mask[i, self.batch_size + i] = 0
            self.mask[self.batch_size + i, i] = 0
        
        z_i= F.normalize(z_i, dim=1)
        z_j= F.normalize(z_j, dim=1)
        
        p1 = torch.cat((z_i, z_j), dim=0)
        
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature
        
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return(loss)


class SimCLRTraining:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self._set_seed()
        self._read_config()
        self._prepare_directories()
        self.history = {'train': [], 'val': []}
        self.best_val = float('inf')
        self.sampled_epochs = []
        self.mt = 0
        
    def _set_seed(self):
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        
    def _read_config(self,file_path='config.json'):
        with open(file_path, 'r') as file:
            config = json.load(file)
        for key, value in config.items():
            setattr(self, key ,value)
        return config
    
    def _prepare_directories(self):
        self.root_dir = Path(self.base_path)
        self.save_dir = self.root_dir / 'self_supervised' / 'simclr'
        self.save_dir_ckpts = self.save_dir / 'simclr_ckpts'
        self.save_dir_mcmc = self.save_dir_ckpts / 'mcmc_samples' / f'{self.args.optimizer}_{self.args.exp}'
        self.save_dir_param = self.save_dir / 'params'
        for path in [self.save_dir,self.save_dir_ckpts,self.save_dir_mcmc, self.save_dir_param]:
            os.makedirs(path, exist_ok=True)
       
        self._save_hyperparameters()
    
    def _save_hyperparameters(self):
        HPS = vars(self.args)
        with open(self.save_dir_param / f'{self.args.exp}_{self.args.model_type}param.json', 'w') as file:
            json.dump(HPS, file, indent=4)
            
    def _load_dataloader(self):
        self.train_loader, self.val_loader = get_aptos_loader(
            base_path=self.base_path,
            size=self.size,
            split_seed=self.args.seed,
            train_size=1-self.args.val_size,
            val_size=self.args.val_size,
            aug=self.args.aug
        )
               
    def _initialize_model_and_optimizer(self):
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)
        self.online_network = SimCLR(backbone, hid_dim=HID_DIM, out_dim=PROJECTION_DIM).to(self.device)
        
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.online_network.parameters(), self.args.lr, weight_decay=self.args.wd)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.online_network.parameters(), lr=self.args.lr, weight_decay=self.args.wd,
                                       momentum=0.9)
        elif self.args.optimizer == 'sghm':
            self.optimizer = SGHM(
                params=self.online_network.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                momentum=0.9,
                temp=0.1,
                N_train=len(self.train_loader.dataset)
            )
        self.criterion = NT_Xent(temperature=0.1, device=self.device)
        
    def train(self):
        
        self._load_dataloader()
        self._initialize_model_and_optimizer()
        
        cycle_batch_length = self.args.cycle_length * len(self.train_loader)
        batch_idx = 0
        self.best_val = float('inf')
        
        print('Training started...')
        for epoch in range(self.args.num_epochs):
            tic = time.time()
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.online_network.train()
                    dataloader = self.train_loader
                else:
                    self.online_network.eval()
                    dataloader = self.val_loader
                
                total_loss = 0
                for (img1, img2) in dataloader:
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    proj1 = self.online_network(img1)
                    proj2 = self.online_network(img2)
                    loss = self._train_step(proj1, proj2, phase, batch_idx, cycle_batch_length, epoch)
                    del proj1, proj2
                    torch.cuda.empty_cache()
                    total_loss += loss.item()
                    if phase == 'train':
                        batch_idx += 1
                
                self.history[phase].append(total_loss / len(dataloader.dataset))
                if phase == 'train':
                    print(f"Epoch {epoch}, Train Loss: {self.history[phase][-1]:.4f}")
                elif phase == 'val':
                    print(f"Epoch {epoch}, Val Loss: {self.history[phase][-1]:.4f}")
            
            self._post_epoch_operations(epoch)
            toc = time.time()
            print(f"Epoch {epoch} completed in {toc - tic:.2f}s")

    def _train_step(self, proj1, proj2, phase, batch_idx, cycle_batch_length, epoch):
        loss = self.criterion(proj1, proj2)
        if phase == 'train':
            self.optimizer.zero_grad()
            CycCosineSch(self.args.lr, batch_idx, cycle_batch_length, self.args.n_sam_cycle, self.optimizer)
            if self.args.epoch_noise and (epoch % self.args.cycle_length) + 1 > self.args.epoch_noise:
                self.optimizer.param_groups[0]['epoch_noise'] = True
            else:
                self.optimizer.param_groups[0]['epoch_noise'] = False
            loss.backward()
            self.optimizer.step()
        return loss
    
    def _post_epoch_operations(self, epoch):
        if self.history['val'][-1] < self.best_val:
            self.best_val = self.history['val'][-1]
            torch.save({
                'epoch': epoch + 1,
                'model': self.online_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'Loss': self.history['val'][-1]
            }, os.path.join(self.save_dir_ckpts, f'{self.args.exp}_best_{self.args.model_type}sclr.pt'))
        
        if self.args.save_sample and epoch >= self.args.epoch_st and \
                (epoch % self.args.cycle_length) + 1 > (self.args.cycle_length - self.args.n_sam_cycle):
            self.sampled_epochs.append(epoch)
            torch.save(self.online_network.state_dict(), os.path.join(self.save_dir_mcmc, f'model_{self.mt}.pt'))
            self.mt += 1
            print(f'Sample {self.mt}/{self.args.N_samples} taken at epoch {epoch}')
        
        torch.save({
            'epoch': epoch + 1,
            'model': self.online_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'Loss': self.history['val'][-1]
        }, os.path.join(self.save_dir_ckpts, f'{self.args.exp}_last_{self.args.model_type}sclr.pt'))
        
        plotCurves(self.history,os.path.join(self.save_dir_param,f'exp_{args.exp}_{args.model_type}loss.png')) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR pretraining')
    parser.add_argument('--seed', type=int, default=42, help='The seed for experiments')
    parser.add_argument('--exp', type=int, default=4, help='ID of this experiment')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for pretraining')
    parser.add_argument('--model_type', type=str, default='cb', choices=['', 'cb'], help='Type of model')
    parser.add_argument('--optimizer', type=str, default='sghm', choices=['adam', 'sgd', 'sghm'], help='Optimizer to use')
    parser.add_argument('--aug', type=str, default='simclr', choices=['simclr', 'contig'], help='Data augmentation type')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation size')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--epoch_noise', type=int, default=40, help='Epoch to inject noise')
    parser.add_argument('--save_sample', type=bool, default=True, help='Save samples or not')
    parser.add_argument('--epoch_st', type=int, default=0, help='Epoch to start saving checkpoints')
    parser.add_argument('--cycle_length', type=int, default=50, help='Cycle length')
    parser.add_argument('--n_sam_cycle', type=int, default=1, help='Number of samples per cycle')
    parser.add_argument('--N_samples', type=int, default=4, help='Total number of samples')
    args = parser.parse_args()
    
    trainer = SimCLRTraining(args)
    trainer.train()
