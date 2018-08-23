import os
import numpy as np
import cv2 as cv
from conversion import image_to_matrix
from preprocess import Convert

def LoadCamVid(path):
    files = os.listdir(path)
    imgfiles = []
    lblfiles = []
    for f in files:
        if ".png" in f:
            if "_L" in f:
                lblfiles.append(os.path.join(path,f))
            else:
                imgfiles.append(os.path.join(path,f))
    assert len(imgfiles) == len(lblfiles)
    return sorted(imgfiles),sorted(lblfiles)

if __name__ == "__main__":
    import argparse
    from classes import getClasses
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir',type=str,default='CamVid/',help='camvid directory')
    parser.add_argument('-classes',type=str,default='CamVid/labels.txt',help='classes file')
    parser.add_argument('-images',type=str,default='Dataset/Images/',help='images path')
    parser.add_argument('-masks',type=str,default='Dataset/Masks/',help='masks path')
    parser.add_argument('-labels',type=str,default='Dataset/vLabels/',help='labels path')
    parser.add_argument('-weights',type=str,default='Dataset/weights.npy',help='weights of each class')
    parser.add_argument('-fx',type=float,default=0.5,help='output size in x axis')
    parser.add_argument('-fy',type=float,default=0.5,help='output size in y axis')
    args = parser.parse_args()

    images,labels = LoadCamVid(args.dir)
    classes = getClasses(args.classes)
    for directory in [args.images,args.masks,args.labels]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    Convert(images,labels,classes,args.images,args.labels,args.masks,args.weights,args.fx,args.fy)