import numpy as np
import cv2 as cv
from classes import getClasses,bgr_numpy

def image_to_matrix(image,classes):
    rows,columns,channels = image.shape
    matrix = np.zeros((rows,columns),dtype=np.int)
    for row in range(rows):
        for col in range(columns):
            converted = False
            for i in range(len(classes)):
                matched = True
                for c in range(channels):
                    if image[row,col,c] != classes[i][1][c]:
                        matched = False
                        break
                if matched:
                    matrix[row,col] = i
                    converted = True
                    break
            if not converted:
                print((row,col))
                print(image[row,col,:])
                assert converted
    return matrix

def matrix_to_image(matrix,classes):
    rows,columns = matrix.shape
    channels = len(classes[0][1])
    image = np.zeros((rows,columns,channels),dtype=np.int)
    for row in range(rows):
        for col in range(columns):
            for c in range(channels):
                image[row,col,c] = classes[matrix[row,col]][1][c]
    return image