import cv2
import numpy
import glob, os

# ========== FUNCTIONS FOR IMAGE RETRIEVAL ==========

def get_all_data_files():
    return list(glob.glob("DATA/*.data.jpg"))

def get_all_test_files():
    return list(glob.glob("TEST/*.data.jpg"))

# ========== FUNCTIONS FOR LABEL RETRIEVAL ==========

# Note: Labels in MNIST were represented as
# [[ 0.  0.  0. ...,  1.  0.  0.]
#  [ 0.  0.  0. ...,  0.  0.  0.]
#  [ 0.  0.  0. ...,  0.  0.  0.]
#   ..., 
#  [ 0.  0.  0. ...,  0.  0.  0.]
#  [ 0.  0.  0. ...,  0.  0.  0.]
#  [ 0.  0.  0. ...,  0.  1.  0.]]

def get_all_data_labels():
    pass

def edge_detect(picture):
    img = cv2.imread(picture, 0) # Load the image in grayscale
    img = cv2.resize(image, (320, 240))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(img, 100, 200, 5)
    return edges

def mat2array(mat):
    return numpy.asarray(mat)

