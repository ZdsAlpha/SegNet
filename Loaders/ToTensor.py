import torch
from torch.utils.data import Dataset

class ToByteTensor(Dataset):
    def __init__(self,loader,device=0):
        self.loader = loader
        self.device = device
    
    def __len__(self):
        return self.loader.__len__()

    def __getitem__(self,index):
        return torch.ByteTensor(self.loader.__getitem__(index)).to(self.device)

class ToLongTensor(Dataset):
    def __init__(self,loader,device=0):
        self.loader = loader
        self.device = device
    
    def __len__(self):
        return self.loader.__len__()

    def __getitem__(self,index):
        return torch.FloatTensor(self.loader.__getitem__(index)).to(self.device)

class ToFloatTensor(Dataset):
    def __init__(self,loader,device=0):
        self.loader = loader
        self.device = device
    
    def __len__(self):
        return self.loader.__len__()

    def __getitem__(self,index):
        return torch.FloatTensor(self.loader.__getitem__(index)).to(self.device)
