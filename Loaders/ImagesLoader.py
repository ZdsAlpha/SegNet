import numpy as np
import cv2 as cv
from Loaders.functions import GetFiles
from torch.utils.data import Dataset

class ImagesLoader(Dataset):
    def __init__(self,directory,pattren='*.png',flag=cv.IMREAD_COLOR):
        self.directory = directory
        self.images = GetFiles(directory,pattren)
        self.flag = flag

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        name = self.images[index]
        image = cv.imread(name,self.flag)
        if len(image.shape) == 2:
            image = np.expand_dims(image,axis=2)
        return image