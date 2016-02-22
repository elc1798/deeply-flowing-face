import cv2
import numpy
import glob, os

def get_all_data_files():
    return list(glob.glob("DATA/*.data.jpg"))

def edge_detect(picture):
    img = cv2.imread(picture, 0) # Load the image in grayscale
    img = cv2.resize(image, (320, 240))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(img, 100, 200, 5)
    return edges

def mat2array(mat):
    return numpy.asarray(mat)

