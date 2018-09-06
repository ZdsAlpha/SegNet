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
    from Loaders.NumpyLoader import NumpyLoader
    from Loaders.Concatinators import Numpy
    from functions import matrices_to_images
    parser = argparse.ArgumentParser()
    parser.add_argument('-masks',type=str,default='Contents/__Dataset/Masks/',help='masks directory')
    parser.add_argument('-classes',type=str,default='Contents/__Dataset/classes.txt',help='classes file')
    parser.add_argument('-device',type=int,default=0,help='device id')
    parser.add_argument('-model',type=str,default='Contents/Models/reconstruct.model',help='path of segnet model')
    parser.add_argument('-batch',type=int,default=4,help='batch size')
    parser.add_argument('-output',type=str,default="Contents/__Output/",help='output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    loader = NumpyLoader(args.masks)
    loader = DataLoader(loader,args.batch,False,num_workers=4)

    def extend_matrix(matrix,device):
        tensors = []
        for i in range(len(classes)):
            tensors.append(torch.eq(matrix,float(i)))
        return torch.stack(tensors,dim=1)

    classes = getClasses(args.classes)
    model = torch.load(args.model)
    index = 0
    def onTestBatch(batch_id,features,output):
        global index
        for img in output:
            img = img.cpu().numpy()
            img = np.transpose(img,(1,2,0)) * 255
            name = os.path.join(args.output,str(index).rjust(4,'0')+'.png')
            cv.imwrite(name,img)
            index += 1
    eval(model,loader,args.device,features_transform=extend_matrix,onBatch=onTestBatch)