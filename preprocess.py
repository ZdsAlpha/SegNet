import os
import numpy as np
import cv2 as cv
from classes import getClasses,bgr_numpy
from conversion import image_to_matrix

def load_classes(file):
    return bgr_numpy(getClasses(file))

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

    