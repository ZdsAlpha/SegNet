import os
import numpy as np
import cv2 as cv
from conversion import weihgted_image_to_matrix

def Convert(images,labels,classes,img_output_dir,lbl_output_dir,plbl_output_dir,weights_output,FX=0.5,FY=0.5):
    assert len(images) == len(labels)
    count = len(images)
    weights = np.zeros((len(classes)),dtype=np.float)
    for i in range(count):
        imagename = images[i]
        labelname = labels[i]
        #Processing image
        image = cv.imread(imagename,cv.IMREAD_COLOR)
        if FX != 1 or FY != 1:
            image = cv.resize(image,(0,0),fx=FX,fy=FY,interpolation = cv.INTER_NEAREST)
        outputname = os.path.join(img_output_dir,str(i).rjust(4,'0')+'.png')
        cv.imwrite(outputname,image)
        #Processing label
        label = cv.imread(labelname,cv.IMREAD_COLOR)
        if FX != 1 or FY != 1:
            label = cv.resize(label,(0,0),fx=FX,fy=FY,interpolation = cv.INTER_NEAREST)
        outputname = os.path.join(lbl_output_dir,str(i).rjust(4,'0')+'.png')
        cv.imwrite(outputname,label)
        #Labels to numpy
        matrix, _weights = weihgted_image_to_matrix(label,classes)
        outputname = os.path.join(plbl_output_dir,str(i).rjust(4,'0')+'.npy')
        np.save(outputname,matrix)
        weights += _weights 
        print("Processed " + str(i+1) + " out of " + str(count))
    print(weights)
    np.save(weights_output,weights)

if __name__ == "__main__":
    import argparse
    from classes import getClasses
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_images',type=str,default='Images/',help='input images directory')
    parser.add_argument('-in_masks',type=str,default='Masks/',help='input masks directory')
    parser.add_argument('-classes',type=str,default='labels.txt',help='classes file')
    parser.add_argument('-images',type=str,default='Dataset/Images/',help='images path')
    parser.add_argument('-masks',type=str,default='Dataset/Masks/',help='masks path')
    parser.add_argument('-labels',type=str,default='Dataset/vLabels/',help='labels path')
    parser.add_argument('-weights',type=str,default='Dataset/weights.npy',help='weights of each class')
    parser.add_argument('-fx',type=float,default=0.5,help='output size in x axis')
    parser.add_argument('-fy',type=float,default=0.5,help='output size in y axis')
    args = parser.parse_args()

    images,labels = sorted(os.listdir(args.in_images)),sorted(os.listdir(args.in_masks))
    for i in range(len(images)):
        images[i] = os.path.join(args.in_images,images[i])
        labels[i] = os.path.join(args.in_masks,labels[i])

    assert len(images) == len(labels)
    classes = getClasses(args.classes)
    for directory in [args.images,args.masks,args.labels]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    Convert(images,labels,classes,args.images,args.labels,args.masks,args.weights,args.fx,args.fy)
    