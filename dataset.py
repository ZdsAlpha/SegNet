import os
import numpy as np
import torch
import glob
import cv2 as cv
import random
import math
from torch.utils.data import Dataset

class SegNetLoader(Dataset):
    def __init__(self,images_dir,labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        images = glob.glob(os.path.join(images_dir,"*.png"))
        labels = glob.glob(os.path.join(labels_dir,"*.npy"))
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image_name = self.images[index]
        label_name = self.labels[index]
        image = cv.imread(image_name,cv.IMREAD_COLOR)
        label = np.load(label_name)
        image = image / 255.0
        image = np.transpose(image,(2,0,1))
        return (torch.FloatTensor(image),torch.LongTensor(label))

class DatasetSelector(Dataset):
    def __init__(self,dataset,ids):
        self.ids = ids
        self.dataset = dataset
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        return self.dataset.__getitem__(self.ids[index])

def divide_dataset(dataset,ratio=0.7,seed=0,shuffle=False):
    ids = list(range(len(dataset)))
    if shuffle:
        random.seed(seed)
        random.shuffle(ids)
    first = ids[:math.floor((len(dataset)-1)*ratio)]
    second = ids[len(first):]
    return (DatasetSelector(dataset,first),DatasetSelector(dataset,second))
