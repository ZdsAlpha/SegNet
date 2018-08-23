import torch

def test(model,dataloader,criterion,device=None,callback=None):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for images,labels in dataloader:
            if device is not None:
                images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss = criterion(output,labels)
            total_loss += loss.item()
            if callback is not None:
                callback(output,loss)
    return total_loss

def run(model,dataloader,device=None,callback=None):
    model.eval()
    with torch.no_grad():
        for images,labels in dataloader:
            if device is not None:
                images,labels = images.to(device),labels.to(device)
            output = model(images)
            if callback is not None:
                callback(output)

if __name__ == "__main__":
    import argparse
    import os
    import cv2 as cv
    from dataset import SegNetLoader
    from classes import getClasses
    from model import SegNet,UNet
    from torch.utils.data import DataLoader
    from conversion import image_to_matrix,matrix_to_image

    parser = argparse.ArgumentParser()
    parser.add_argument('-images',type=str,default='Dataset/Images/',help='images directory')
    parser.add_argument('-masks',type=str,default='Dataset/Masks/',help='masks directory')
    parser.add_argument('-classes',type=str,default='CamVid/labels.txt',help='classes file')
    parser.add_argument('-device',type=int,default=0,help='device id')
    parser.add_argument('-model',type=str,default='segnet.model',help='path of segnet model')
    parser.add_argument('-batch',type=int,default=4,help='batch size')
    parser.add_argument('-output',type=str,default="Output/",help='output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    dataset = SegNetLoader(args.images,args.masks)
    classes = getClasses(args.classes)
    model = torch.load(args.model).to(args.device)
    loader = DataLoader(dataset,args.batch,shuffle=False)
    index = 0
    def onTestBatch(output):
        global index
        output=torch.argmax(output,dim=1)
        for img in output:
            img = matrix_to_image(img.cpu().numpy(),classes)
            name = os.path.join(args.output,str(index).rjust(4,'0')+'.png')
            cv.imwrite(name,img)
            print(name)
            index += 1

    run(model,loader,args.device,onTestBatch)
    
