import torch
from torch.utils.data import Dataset

class Loaders(Dataset):
    def __init__(self,loaders):
        length = len(loaders[0])
        for loader in loaders:
            assert len(loader) == length
        self.loaders = loaders
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        items = [None] * len(self.loaders)
        for i in range(len(self.loaders)):
            items[i] = self.loaders[i].__getitem__(index)
        return tuple(items)

class Tensors(Dataset):
    def __init__(self,*loaders,dim=0):
        length = len(loaders[0])
        for loader in loaders:
            assert len(loader) == length
        self.loaders = loaders
        self.length = length
        self.dim = dim
    
    def __len__(self):
        return self.length

    def __getitem__(self,index):
        tensors = [None] * len(self.loaders)
        for i in range(len(self.loaders)):
            tensors[i] = self.loaders[i].__getitem__(index)
        return torch.cat(tensors,dim=self.dim)