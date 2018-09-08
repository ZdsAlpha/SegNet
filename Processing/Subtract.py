if __name__ == "__main__":
    import os
    import numpy as np
    import cv2 as cv
    import argparse
    from Loaders.ImagesLoader import ImagesLoader
    from Loaders.ImageResizer import ImageResizer
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_images',type=str,default='Contents/Images/',help='input images directory')
    parser.add_argument('-alpha',type=str,default='Contents/Dataset/Sub/',help='images path')
    parser.add_argument('-fx',type=float,default=1,help='output size in x axis')
    parser.add_argument('-fy',type=float,default=1,help='output size in y axis')
    args = parser.parse_args()
    for directory in [args.alpha]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    images_loader = ImagesLoader(args.in_images)
    images_loader = ImageResizer(images_loader,args.fx,args.fy)
    last_image = None
    index = 0
    for image in images_loader:
        image = image / 255
        if last_image is None:
            last_image = image
            image = np.zeros(image.shape)
        else:
            b,g,r = image[:,:,0],image[:,:,1],image[:,:,2]
            lb,lg,lr = last_image[:,:,0],last_image[:,:,1],last_image[:,:,2]
            last_image = image
            image = (np.absolute(lb-b) + np.absolute(lb-g) + np.absolute(lr-r)) / 3
        image = image * 255
        outputname = os.path.join(args.alpha,str(index).rjust(4,'0')+'.png')
        cv.imwrite(outputname,image)
        index += 1
        print("Processed " + str(index) + " out of " + str(len(images_loader)))