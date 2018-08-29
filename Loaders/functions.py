import os
import glob
import math
import random
from Loaders.Selector import Selector
from Loaders.Concatinators import Loaders

def GetFiles(directory,pattren='*'):
    return sorted(glob.glob(os.path.join(directory,pattren)))

def divide_dataset(dataset,ratio=0.7,seed=0,shuffle=False):
    ids = list(range(len(dataset)))
    if shuffle:
        random.seed(seed)
        random.shuffle(ids)
    first = ids[:math.floor((len(dataset)-1)*ratio)]
    second = ids[len(first):]
    return (Selector(dataset,first),Selector(dataset,second))

def concat_loaders(*loaders):
    return Loaders(loaders)
