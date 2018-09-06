if __name__ == '__main__':
    import os
    import torch
    import argparse
    import cv2 as cv
    import numpy as np
    from train import train
    from test import test
    from Models.UNet import UNet
    from functions import getClasses,accuracy
    from torch.utils.data import DataLoader
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageConverter import ImageConverter
    from Loaders.NumpyLoader import NumpyLoader
    from Loaders.Concatinators import Loaders,Numpy
    from Loaders.functions import divide_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('-images',type=str,default='Contents/__Dataset/Images/',help='images directory')
    parser.add_argument('-masks',type=str,default='Contents/__Dataset/Masks/',help='masks directory')
    parser.add_argument('-classes',type=str,default='Contents/__Dataset/classes.txt',help='classes file')
    parser.add_argument('-ratio',type=float,default=0.75,help='ratio between train and test dataset')
    parser.add_argument('-seed',type=int,default=1,help='seed value for dividing dataset')
    parser.add_argument('-device',type=int,default=0,help='device id')
    parser.add_argument('-depth',type=int,default=7,help='depth of segnet model')
    parser.add_argument('-filters',type=int,default=16,help='number of filters in first layer')
    parser.add_argument('-model',type=str,default='Contents/Models/reconstruct.model',help='path of segnet model')
    parser.add_argument('-epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('-batch',type=int,default=2,help='batch size')
    parser.add_argument('-lr',type=float,default=0.0001,help='learning rate')
    args = parser.parse_args()

    images_loader = ImagesLoader(args.images)
    images_loader = ImageConverter(images_loader)
    mask_loader = NumpyLoader(args.masks)
    loader = Loaders((mask_loader,images_loader))
    train_loader,test_loader = divide_dataset(loader,ratio=args.ratio,seed=args.seed)
    train_loader = DataLoader(train_loader,args.batch,True,num_workers=4)
    test_loader = DataLoader(test_loader,args.batch,False,num_workers=4)
    classes = getClasses(args.classes)

    def extend_matrix(matrix,device):
        tensors = []
        for i in range(len(classes)):
            tensors.append(torch.eq(matrix,float(i)))
        return torch.stack(tensors,dim=1)
    
    if os.path.isfile(args.model):
        model = torch.load(args.model).to(args.device)
        print("Model loaded!")
    else:
        model = UNet(len(classes),3,args.depth,args.filters).to(args.device)
        model.initialize()
        print("Model initialized!")
    print(model)
    criterion = torch.nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),args.lr)
    print("Training model...")

    min_loss = float('inf')
    def onTrainEpoch(epoch,loss):
        global min_loss
        print("Epoch #" + str(epoch) + " Train Loss: " + str(loss))
        loss = test(model,test_loader,criterion,args.device,features_transform=extend_matrix,labels_type=torch.FloatTensor)
        print("Epoch #" + str(epoch) + " Test Loss: " + str(loss))
        if loss < min_loss:
            print("Saving model...")
            torch.save(model,args.model)
            print("Model saved!")
            min_loss = loss

    total_loss = train(model,train_loader,criterion,optimizer,args.epochs,args.device,features_transform=extend_matrix,onEpoch=onTrainEpoch,labels_type=torch.FloatTensor)
    print("Training finished!")
    print("Average train loss: " + str(total_loss))