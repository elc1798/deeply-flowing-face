import cv2
import numpy
import glob, os

# ========== FUNCTIONS FOR IMAGE PROCESSING ==========

def edge_detect(picture):
    """
    Runs Canny edge detection on a OpenCV Mat

    Params:
        picture - An OpenCV Math to process

    Returns:
        The Canny edges detected in `picture`
    """
    img = cv2.imread(picture, 0) # Load the image in grayscale
    img = cv2.resize(image, (320, 240))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(img, 100, 200, 5)
    return edges

def mat2array(mat):
    """
    Converts an OpenCV Mat to a NumPy array

    Params:
        mat - An OpenCV Mat

    Returns:
        The NumPy array representation of `mat`
    """
    return numpy.asarray(mat)

# ========== FUNCTIONS FOR IMAGE RETRIEVAL ==========

def get_all_data_files():
    """
    Returns a list of all the .data.jpg files in DATA/
    """
    return list(glob.glob("DATA/*.data.jpg"))

def get_all_test_files():
    """
    Returns a list of all the .data.jpg files in TEST/
    """
    return list(glob.glob("TEST/*.data.jpg"))

def get_all_input_files():
    """
    Returns a list of all the .input.png files in INPUTS/
    """
    return list(glob.glob("INPUTS/*.input.png"))

# ========== FUNCTIONS FOR DATA RETRIEVAL ==========

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

