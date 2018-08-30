import numpy as np
from Loaders.functions import GetFiles
from torch.utils.data import Dataset

class NumpyLoader(Dataset):
    def __init__(self,directory,pattren='*.npy'):
        self.directory = directory
        self.objects = GetFiles(directory,pattren)
        self.pattren = pattren
    
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self,index):
        name = self.objects[index]
        obj = np.load(name)
        return obj