import numpy as np
from torch.utils.data import Dataset

class ImageConverter(Dataset):
    def __init__(self,images_loader):
        self.loader = images_loader

    def __len__(self):
        return self.__len__()

    def __getitem__(self,index):
        image = self.__getitem__(index)
        image = image / 255.0
        indices = list(range(len(image.shape)))
        indices = indices[-1]+indices[:-1]
        image = np.transpose(image,indices)
        return image