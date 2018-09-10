if __name__ == '__main__':
    import os
    import torch
    import argparse
    import cv2 as cv
    from test import eval
    from functions import getClasses
    from torch.utils.data import DataLoader
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageConverter import ImageConverter
    from Loaders.Concatinators import Numpy
    from functions import matrices_to_images
    from Loaders.functions import concat_loaders
    parser = argparse.ArgumentParser()
    parser.add_argument('-in1',type=str,default='Contents/Dataset/vLabels/',help='first input mask')
    parser.add_argument('-in2',type=str,default='Contents/Dataset/Alpha/',help='second input mask')
    parser.add_argument('-out',type=str,default='Contents/Dataset/AND/',help='output')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.out)
    masks1 = ImagesLoader(args.in1,flag=cv.IMREAD_GRAYSCALE)
    masks1 = ImageConverter(masks1)
    masks2 = ImagesLoader(args.in2,flag=cv.IMREAD_GRAYSCALE)
    masks2 = ImageConverter(masks2)
    masks = concat_loaders(masks1,masks2)
    index = 0
    for mask1,mask2 in masks:
        mask = cv.bitwise_and(mask1,mask2)
        outputname = os.path.join(args.alpha,str(index).rjust(4,'0')+'.png')
        index += 1
