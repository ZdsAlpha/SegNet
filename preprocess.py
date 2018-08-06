import os
import numpy as np
import cv2 as cv
from conversion import image_to_matrix

def Convert(images,labels,classes,img_output_dir,lbl_output_dir,plbl_output_dir,FX=0.5,FY=0.5):
    assert len(images) == len(labels)
    count = len(images)
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
        matrix = image_to_matrix(label,classes)
        outputname = os.path.join(plbl_output_dir,str(i).rjust(4,'0')+'.npy')
        np.save(outputname,matrix)
        print("Processed " + str(i+1) + " out of " + str(count))

def LoadCamVid(path):
    files = os.listdir(path)
    imgfiles = []
    lblfiles = []
    i = 0
    for f in range(len(files)):
        if '.png' in files[f]:
            if f % 2 == 0:
                imgfiles.append(os.path.join(path,files[i]))
            else:
                lblfiles.append(os.path.join(path,files[i]))
            i += 1
    assert len(imgfiles) == len(lblfiles)
    return imgfiles,lblfiles

if __name__ == "__main__":
    import argparse
    from classes import getClasses
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir',type=str,default='CamVid/',help='camvid directory')
    parser.add_argument('-classes',type=str,default='CamVid/label_colors.txt',help='classes file')
    parser.add_argument('-images',type=str,default='Dataset/Images/',help='images path')
    parser.add_argument('-masks',type=str,default='Dataset/Masks/',help='masks path')
    parser.add_argument('-labels',type=str,default='Dataset/vLabels/',help='labels path')
    parser.add_argument('-fx',type=float,default=0.5,help='output size in x axis')
    parser.add_argument('-fy',type=float,default=0.5,help='output size in y axis')
    args = parser.parse_args()

    images,labels = LoadCamVid(args.dir)
    classes = getClasses(args.classes)
    for directory in [args.images,args.masks,args.labels]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    Convert(images,labels,classes,args.images,args.labels,args.masks,args.fx,args.fy)
    