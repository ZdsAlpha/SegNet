import torch

def train(model,datalaoder,criterion,optimizer,epochs=1,device=None,callback=None):
    loss_sum=0
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for images,labels in datalaoder:
            if device is not None:
                images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del images,labels,output
        if callback is not None:
            callback(epoch+1,total_loss)
    return loss_sum / epochs

if __name__ == "__main__":
    import argparse
    import os
    from test import test
    from dataset import SegNetLoader,SegNetLoader3D,divide_dataset
    from classes import getClasses
    from model import SegNet3D
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('-images',type=str,default='Dataset/Images/',help='images directory')
    parser.add_argument('-masks',type=str,default='Dataset/Masks/',help='masks directory')
    parser.add_argument('-ratio',type=float,default=0.8,help='ratio between train and test dataset')
    parser.add_argument('-seed',type=int,default=1,help='seed for dividing dataset')
    parser.add_argument('-classes',type=str,default='CamVid/label_colors.txt',help='classes file')
    parser.add_argument('-device',type=int,default=0,help='device id')
    parser.add_argument('-depth',type=int,default=2,help='depth of segnet model')
    parser.add_argument('-filters',type=int,default=32,help='number of filters in first layer')
    parser.add_argument('-model',type=str,default='segnet3d.model',help='path of segnet model')
    parser.add_argument('-epochs',type=int,default=50,help='number of epochs')
    parser.add_argument('-batch',type=int,default=1,help='batch size')
    parser.add_argument('-frames',type=int,default=4,help='number of frames')
    parser.add_argument('-lr',type=float,default=0.001,help='learning rate')
    args = parser.parse_args()

    dataset = SegNetLoader3D(args.images,args.masks,args.frames)
    train_dataset,test_dataset = divide_dataset(dataset,args.ratio,args.seed,True)
    classes = getClasses(args.classes)
    if os.path.isfile(args.model):
        model = torch.load(args.model).to(args.device)
        print("Model loaded!")
    else:
        model = SegNet3D(3,len(classes),args.frames,args.depth,args.filters).to(args.device)
        model.initialize()
        print("Model initialized!")
    print(model)
    train_loader = DataLoader(train_dataset,args.batch,True,num_workers=4)
    test_loader = DataLoader(test_dataset,args.batch,False,num_workers=4)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
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