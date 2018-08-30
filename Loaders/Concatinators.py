import torch
import numpy as np
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

class Numpy(Dataset):
    def __init__(self,loaders,axis=0):
        length = len(loaders[0])
        for loader in loaders:
            assert len(loader) == length
        self.loaders = loaders
        self.length = length
        self.axis = axis
    
    def __len__(self):
        return self.length 
    
    def __getitem__(self,index):
        arrays = [None] * len(self.loaders)
        for i in range(len(self.loaders)):
            arrays[i] = self.loaders[i].__getitem__(index)
        return np.concatenate(arrays,axis=self.axis)

class Tensors(Dataset):
    def __init__(self,loaders,dim=0):
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