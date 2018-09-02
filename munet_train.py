if __name__ == '__main__':
    import os
    import torch
    import argparse
    import cv2 as cv
    import numpy as np
    from train import train
    from test import test
    from Models.UNet import UNet
    from functions import getClasses
    from torch.utils.data import DataLoader
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageConverter import ImageConverter
    from Loaders.NumpyLoader import NumpyLoader
    from Loaders.Concatinators import Loaders,Numpy
    from Loaders.functions import divide_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('-images',type=str,default='Contents/Dataset/Images/',help='images directory')
    parser.add_argument('-alpha',type=str,default='Contents/Dataset/Alpha/',help='alpha channel for images')
    parser.add_argument('-masks',type=str,default='Contents/Dataset/Masks/',help='masks directory')
    parser.add_argument('-classes',type=str,default='Contents/Dataset/classes.txt',help='classes file')
    parser.add_argument('-weights',type=str,default='Contents/Dataset/weights.npy',help='weights for cross-entropy loss')
    parser.add_argument('-ratio',type=float,default=0.75,help='ratio between train and test dataset')
    parser.add_argument('-seed',type=int,default=1,help='seed value for dividing dataset')
    parser.add_argument('-device',type=int,default=0,help='device id')
    parser.add_argument('-depth',type=int,default=5,help='depth of segnet model')
    parser.add_argument('-filters',type=int,default=4,help='number of filters in first layer')
    parser.add_argument('-model',type=str,default='Output/unet_masked.model',help='path of segnet model')
    parser.add_argument('-epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('-batch',type=int,default=2,help='batch size')
    parser.add_argument('-lr',type=float,default=0.001,help='learning rate')
    args = parser.parse_args()

    images_loader = ImagesLoader(args.images)
    images_loader = ImageConverter(images_loader)
    alpha_loader = ImagesLoader(args.alpha,flag=cv.IMREAD_GRAYSCALE)
    alpha_loader = ImageConverter(alpha_loader)
    input_loader = Numpy((images_loader,alpha_loader))
    mask_loader = NumpyLoader(args.masks)
    loader = Loaders((input_loader,mask_loader))
    train_loader,test_loader = divide_dataset(loader,ratio=args.ratio,seed=args.seed)
    train_loader = DataLoader(train_loader,args.batch,True,num_workers=4)
    test_loader = DataLoader(test_loader,args.batch,False,num_workers=4)
    classes = getClasses(args.classes)
    weights = None
    if os.path.isfile(args.weights):
        weights = np.load(args.weights)
        weights = weights / np.sum(weights)
        weights = 1 - weights
        print(weights)
    if os.path.isfile(args.model):
        model = torch.load(args.model).to(args.device)
        print("Model loaded!")
    else:
        model = UNet(4,len(classes),args.depth,args.filters).to(args.device)
        model.initialize()
        print("Model initialized!")
    print(model)
    if weights is None:
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
    else:
        weights = torch.FloatTensor(weights).to(args.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),args.lr)
    print("Training model...")
    
    min_loss = float('inf')
    def onTrainEpoch(epoch,loss):
        global min_loss
        print("Epoch #" + str(epoch) + " Train Loss: " + str(loss))
        loss = test(model,test_loader,criterion,args.device)
        print("Epoch #" + str(epoch) + " Test Loss: " + str(loss))
        if loss < min_loss:
            print("Saving model...")
            torch.save(model,args.model)
            print("Model saved!")
            min_loss = loss

    total_loss = train(model,train_loader,criterion,optimizer,args.epochs,args.device,onTrainEpoch)
    print("Training finished!")
    print("Average train loss: " + str(total_loss))

