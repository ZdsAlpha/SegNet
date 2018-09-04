if __name__ == '__main__':
    import os
    import torch
    import argparse
    import cv2 as cv
    import numpy as np
    from test import eval
    from Models.UNet import UNet
    from functions import getClasses
    from torch.utils.data import DataLoader
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageConverter import ImageConverter
    from Loaders.Concatinators import Numpy
    from functions import matrices_to_images
    parser = argparse.ArgumentParser()
    parser.add_argument('-images',type=str,default='Contents/Dataset/Images/',help='images directory')
    parser.add_argument('-alpha',type=str,default='Contents/Dataset/Alpha/',help='alpha channel for images')
    parser.add_argument('-classes',type=str,default='Contents/Dataset/classes.txt',help='classes file')
    parser.add_argument('-device',type=int,default=0,help='device id')
    parser.add_argument('-model',type=str,default='Contents/Models/unet_masked.model',help='path of segnet model')
    parser.add_argument('-batch',type=int,default=4,help='batch size')
    parser.add_argument('-output',type=str,default="Contents/Output/",help='output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    images_loader = ImagesLoader(args.images)
    images_loader = ImageConverter(images_loader)
    alpha_loader = ImagesLoader(args.alpha,flag=cv.IMREAD_GRAYSCALE)
    alpha_loader = ImageConverter(alpha_loader)
    loader = Numpy((images_loader,alpha_loader))
    loader = DataLoader(loader,args.batch,False,num_workers=4)
    classes = getClasses(args.classes)
    model = torch.load(args.model)
    index = 0
    def onTestBatch(batch_id,features,output):
        global index
        output = torch.argmax(output,dim=1)
        output = matrices_to_images(output,classes,args.device)
        for img in output:
            img = img.cpu().numpy()
            name = os.path.join(args.output,str(index).rjust(4,'0')+'.png')
            cv.imwrite(name,img)
            index += 1
    eval(model,loader,args.device,onBatch=onTestBatch)