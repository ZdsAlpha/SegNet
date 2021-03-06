if __name__ == "__main__":
    import os
    import numpy as np
    import cv2 as cv
    import argparse
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageResizer import ImageResizer
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_images',type=str,default='Contents/Images/',help='input images directory')
    parser.add_argument('-images',type=str,default='Contents/Dataset/Images/',help='images path')
    parser.add_argument('-fx',type=float,default=1,help='output size in x axis')
    parser.add_argument('-fy',type=float,default=1,help='output size in y axis')
    args = parser.parse_args()
    for directory in [args.images]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    images_loader = ImagesLoader(args.in_images)
    images_loader = ImageResizer(images_loader,args.fx,args.fy)
    index = 0
    for image in images_loader:
        outputname = os.path.join(args.images,str(index).rjust(4,'0')+'.png')
        cv.imwrite(outputname,image)
        index += 1
        print("Processed " + str(index) + " out of " + str(len(images_loader)))