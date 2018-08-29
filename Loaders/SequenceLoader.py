import numpy as np
from torch.utils.data import Dataset

class SequenceLoader(Dataset):
    def __init__(self,loader,frames):
        self.loader = loader
        self.frames= frames
    
    def __len__(self):
        return self.loader.__len__() - self.frames + 1
    
    def __getitem__(self,index):
        array = []
        for i in range(index,index+self.frames):
            obj = self.loader.__getitem__(i)
            array.append(obj)
        return np.stack(array)