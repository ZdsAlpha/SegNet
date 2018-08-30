if __name__ == "__main__":
    import os
    import numpy as np
    import cv2 as cv
    import torch
    import argparse
    from functions import getClasses,images_to_matrices,matrices_to_images
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageResizer import ImageResizer
    from Loaders.functions import concat_loaders
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_images',type=str,default='Contents/Images/',help='input images directory')
    parser.add_argument('-in_masks',type=str,default='Contents/Masks/',help='input masks directory')
    parser.add_argument('-classes',type=str,default='Contents/labels.txt',help='classes file')
    parser.add_argument('-batch',type=int,default=32,help='batch size')
    parser.add_argument('-device',type=int,default=0,help='device')
    parser.add_argument('-images',type=str,default='Contents/Dataset/Images/',help='images path')
    parser.add_argument('-masks',type=str,default='Contents/Dataset/Masks/',help='masks path')
    parser.add_argument('-labels',type=str,default='Contents/Dataset/vLabels/',help='labels path')
    parser.add_argument('-weights',type=str,default='Contents/Dataset/weights.npy',help='weights of each class')
    parser.add_argument('-fx',type=float,default=1,help='output size in x axis')
    parser.add_argument('-fy',type=float,default=1,help='output size in y axis')
    args = parser.parse_args()
    for directory in [args.images,args.masks,args.labels]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    images_loader = ImagesLoader(args.in_images)
    images_loader = ImageResizer(images_loader,args.fx,args.fy)
    masks_loader = ImagesLoader(args.in_masks)
    masks_loader = ImageResizer(masks_loader,args.fx,args.fy)
    loader = concat_loaders(images_loader,masks_loader)
    classes = getClasses(args.classes)
    batch_loader = DataLoader(loader,args.batch,False,num_workers=4)
    weights = np.zeros(len(classes))
    index = 0
    for images,masks in batch_loader:
        images,masks = torch.ByteTensor(images).to(args.device),torch.ByteTensor(masks).to(args.device)
        matrices,_weights = images_to_matrices(masks,classes,args.device)
        masks = matrices_to_images(matrices,classes,args.device)
        weights += _weights
        for i in range(len(images)):
            outputname = os.path.join(args.images,str(index).rjust(4,'0')+'.png')
            cv.imwrite(outputname,images[i].cpu().numpy())
            outputname = os.path.join(args.masks,str(index).rjust(4,'0')+'.npy')
            np.save(outputname,matrices[i].cpu().numpy())
            outputname = os.path.join(args.labels,str(index).rjust(4,'0')+'.png')
            cv.imwrite(outputname,masks[i].cpu().numpy())
            index += 1
            print("Processed " + str(index) + " out of " + str(len(loader)))
    print(weights)
    np.save(args.weights,weights)
    
    