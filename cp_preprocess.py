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
    parser.add_argument('-dir',type=str,default='Contents/Clay Pigeon/')
    parser.add_argument('-classes',type=str,default='Contents/labels.txt',help='classes file')
    parser.add_argument('-batch',type=int,default=16,help='batch size')
    parser.add_argument('-device',type=int,default=0,help='device')
    parser.add_argument('-images',type=str,default='Contents/Dataset/Images/',help='images path')
    parser.add_argument('-masks',type=str,default='Contents/Dataset/Masks/',help='masks path')
    parser.add_argument('-alpha',type=str,default='Contents/Dataset/Alpha/',help='Alpha channel')
    parser.add_argument('-labels',type=str,default='Contents/Dataset/vLabels/',help='labels path')
    parser.add_argument('-weights',type=str,default='Contents/Dataset/weights.npy',help='weights of each class')
    parser.add_argument('-fx',type=float,default=1,help='output size in x axis')
    parser.add_argument('-fy',type=float,default=1,help='output size in y axis')
    args = parser.parse_args()
    for directory in [args.images,args.masks,args.alpha,args.labels]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    directories = []
    for parent_dir in [os.path.join(args.dir, _directory) for _directory in os.listdir(args.dir)]:
        for directory in [os.path.join(parent_dir,_directory) for _directory in os.listdir(parent_dir)]:
            images = os.path.join(directory,'images')
            masks = os.path.join(directory,'masks')
            assert os.path.isdir(images) == os.path.isdir(masks)
            if os.path.isdir(images) and os.path.isdir(masks):
                directories.append((images,masks))
    print(directories)
    classes = getClasses(args.classes)
    weights = np.zeros(len(classes))
    uindex = 0
    for images_dir,labels_dir in directories:
        print((images_dir,labels_dir))
        index = 0
        images_loader = ImagesLoader(images_dir)
        images_loader = ImageResizer(images_loader,args.fx,args.fy)
        masks_loader = ImagesLoader(labels_dir)
        masks_loader = ImageResizer(masks_loader,args.fx,args.fy)
        loader = concat_loaders(images_loader,masks_loader)
        batch_loader = DataLoader(loader,args.batch,False,num_workers=4)
        mog = cv.bgsegm.createBackgroundSubtractorMOG()
        for images,masks in batch_loader:
            masks = torch.ByteTensor(masks).to(args.device)
            matrices,_weights = images_to_matrices(masks,classes,args.device)
            masks = matrices_to_images(matrices,classes,args.device)
            weights += _weights
            for i in range(len(images)):
                image = images[i].cpu().numpy()
                mogimg = mog.apply(image)
                matrix = matrices[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                if index != 0:
                    outputname = os.path.join(args.images,str(uindex).rjust(4,'0')+'.png')
                    cv.imwrite(outputname,image)
                    outputname = os.path.join(args.alpha,str(uindex).rjust(4,'0')+'.png')
                    cv.imwrite(outputname,mogimg)
                    outputname = os.path.join(args.masks,str(uindex).rjust(4,'0')+'.npy')
                    np.save(outputname,matrix)
                    outputname = os.path.join(args.labels,str(uindex).rjust(4,'0')+'.png')
                    cv.imwrite(outputname,mask)
                else:
                    print("Skipped first image in sequence!")
                uindex += 1
                index += 1
                print("Processed " + str(index) + " out of " + str(len(loader)) + ". Total: " + str(uindex))
    print(weights)
    np.save(args.weights,weights)