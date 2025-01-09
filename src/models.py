import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

HID_DIM = 2048
OUT_DIM = 128


#### Renet backbone
class resnet50_fext(nn.Module):
    def __init__(self,pretarin=True):  
        super(resnet50,self).__init__()
        if pretarin:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
    
    def forward(self,x):
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0],-1)
        return(embedding)
    
    
#### Linear model
class MLP(nn.Module): 
    def __init__(self,in_dim,mlp_hid_size,proj_size):
        super(MLP,self).__init__()
        self.head = nn.Sequential(nn.Linear(in_dim,mlp_hid_size),
                                 nn.BatchNorm1d(mlp_hid_size),
                                 nn.ReLU(),
                                 nn.Linear(mlp_hid_size,proj_size))        
    def forward(self,x):
        x= self.head(x)
        return(x)
    
#### Byol model
class BYOL(nn.Module):
    def __init__(self,net,backbone,hid_dim,out_dim):  
        super(BYOL,self).__init__()
        self.net = net
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim= backbone.fc.in_features,mlp_hid_size=hid_dim,proj_size=out_dim) 
        self.prediction = MLP(in_dim= out_dim,mlp_hid_size=hid_dim,proj_size=out_dim)
        
    def forward(self,x):
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0],-1)
        project = self.projection(embedding)
        
        if self.net=='target':
            return(project)
        predict = self.prediction(project)
        return(predict)
    
#### SimCLR model    
class SimCLR(nn.Module):
    def __init__(self,backbone,hid_dim,out_dim):  
        super(SimCLR,self).__init__()
        # we get representations from avg_pooling layer
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.projection = MLP(in_dim= backbone.fc.in_features,mlp_hid_size=hid_dim,proj_size=out_dim) 

    def forward(self,x):        
        embedding = self.encoder(x)
        embedding = embedding.view(embedding.size()[0],-1)
        project = self.projection(embedding)
        return(project)
