if __name__ == '__main__':
    import os
    import torch
    import argparse
    import cv2 as cv
    import numpy as np
    from train import train
    from test import test
    from Models.UNet import UNet
    from functions import getClasses,confusion_matrix,precision,recall,f1
    from torch.utils.data import DataLoader
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageConverter import ImageConverter
    from Loaders.NumpyLoader import NumpyLoader
    from Loaders.Concatinators import Loaders,Numpy
    from Loaders.functions import divide_dataset
    from beautifultable import BeautifulTable

    parser = argparse.ArgumentParser()
    parser.add_argument('-images',type=str,default='Contents/Dataset/Images/',help='images directory')
    parser.add_argument('-alpha',type=str,default='Contents/Dataset/Alpha/',help='alpha channel for images')
    parser.add_argument('-masks',type=str,default='Contents/Dataset/Masks/',help='masks directory')
    parser.add_argument('-classes',type=str,default='Contents/Dataset/classes.txt',help='classes file')
    parser.add_argument('-weights',type=str,default='Contents/Dataset/weights.npy',help='weights for cross-entropy loss')
    parser.add_argument('-ratio',type=float,default=0.75,help='ratio between train and test dataset')
    parser.add_argument('-seed',type=int,default=1,help='seed value for dividing dataset')
    parser.add_argument('-device',type=int,default=0,help='device id')
    parser.add_argument('-depth',type=int,default=5,help='depth of unet model')
    parser.add_argument('-filters',type=int,default=4,help='number of filters in first layer')
    parser.add_argument('-convperlayer',type=int,default=2,help='number of conv layers per encoder/decoder')
    parser.add_argument('-dropout',type=float,default=0,help='dropout rate')
    parser.add_argument('-model',type=str,default='Contents/Models/unet.model',help='path of model')
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
    loaded = False
    if os.path.isfile(args.weights):
        weights = np.load(args.weights)
        weights = weights / np.sum(weights)
        weights = 1 - weights
        print(weights)
    if os.path.isfile(args.model):
        model = torch.load(args.model).to(args.device)
        print("Model loaded!")
        loaded = True
    else:
        model = UNet(4,len(classes),args.depth,args.filters,convPerLayer=args.convperlayer,dropout=args.dropout).to(args.device)
        model.initialize()
        print("Model initialized!")
    if weights is None:
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
    else:
        weights = torch.FloatTensor(weights).to(args.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),args.lr)

    def print_matrix(matrix,classes):
        p = precision(matrix)
        r = recall(matrix)
        f = f1(p,r)
        table = BeautifulTable()
        table.column_headers = ["Class Name","Precision","Recall","F1 Score"]
        for i in range(len(classes)):
            table.append_row([classes[i][0],p[i],r[i],f[i]])
        print(table)
        total_f1 = np.sum(f) / len(f)
        print("Total F1 score: " + str(total_f1))
        return total_f1

    train_matrix = np.zeros((len(classes),len(classes)),dtype=np.int32)
    def onTrainBatch(batch_id,features,labels,output,loss):
        global train_matrix
        output = torch.argmax(output,dim=1)
        mat = confusion_matrix(labels,output,len(classes))
        train_matrix = np.add(train_matrix,mat)

    test_matrix = np.zeros((len(classes),len(classes)))
    def onTestBatch(batch_id,features,labels,output,loss):
        global test_matrix
        output = torch.argmax(output,dim=1)
        mat = confusion_matrix(labels,output,len(classes))
        test_matrix = np.add(test_matrix,mat)

    score = 0
    print("Testing model...")
    loss = test(model,test_loader,criterion,args.device,onBatch=onTestBatch)
    print("Loss: " + str(loss))
    _score = print_matrix(test_matrix,classes)
    if loaded:
        score = _score
    test_matrix = np.zeros((len(classes),len(classes)))

    print("Training model...")
    def onTrainEpoch(epoch,loss):
        global score
        global train_matrix
        global test_matrix
        print("Epoch #" + str(epoch) + " - Train Loss: " + str(loss))
        trian_score = print_matrix(train_matrix,classes)
        loss = test(model,test_loader,criterion,args.device,onBatch=onTestBatch)
        print("Epoch #" + str(epoch) + " - Test Loss: " + str(loss))
        test_score = print_matrix(test_matrix,classes)
        train_matrix = np.zeros((len(classes),len(classes)))
        test_matrix = np.zeros((len(classes),len(classes)))
        if test_score > score:
            print("Saving model...")
            torch.save(model,args.model)
            print("Model saved!")
            score = test_score

    total_loss = train(model,train_loader,criterion,optimizer,args.epochs,args.device,onBatch=onTrainBatch,onEpoch=onTrainEpoch)
    print("Training finished!")