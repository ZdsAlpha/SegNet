import numpy as np
import cv2 as cv
from torch.utils.data import Dataset

class ImageResizer(Dataset):
    def __init__(self,loader,fx=0.5,fy=0.5,interpolation=cv.INTER_NEAREST):
        self.loader = loader
        self.fx = fx
        self.fy = fy
        self.interpolation = interpolation

    def __len__(self):
        return len(self.loader)
    
    def __getitem__(self,index):
        image = self.loader.__getitem__(index)
        if self.fx != 1 or self.fy != 1:
            image = cv.resize(image,(0,0),fx=self.fx,fy=self.fy,interpolation=self.interpolation)
        return image