import cv2
import numpy
import tensorflow as tf
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
    returns a list of all the .input.png files in INPUTS/
    """
    return list(glob.glob("INPUTS/*.input.png"))

def get_all_test_files():
    """
    returns a list of all the .test.png files in TESTS/
    """
    return list(glob.glob("TESTS/*.test.png"))

# ========== FUNCTIONS FOR DATA RETRIEVAL ==========

# Note: Labels in MNIST were represented as
# [[ 0.  0.  0. ...,  1.  0.  0.]
#  [ 0.  0.  0. ...,  0.  0.  0.]
#  [ 0.  0.  0. ...,  0.  0.  0.]
#   ..., 
#  [ 0.  0.  0. ...,  0.  0.  0.]
#  [ 0.  0.  0. ...,  0.  0.  0.]
#  [ 0.  0.  0. ...,  0.  1.  0.]]

def generate_input_name(data_filename):
    """
    Note that data_filename is in the format: NUMBER_LABEL.data.jpg. This
    function takes such a string and returns the string for the input file.

    Params:
        data_filename - name of a data file

    Returns:
        The corresponding name of the input file for provided data file, as a
        string.
    """
    name = data_filename[5:].split(".data.jpg")[0]
    return "INPUTS/" + name + ".input.png"

def generate_test_name(data_filename):
    """
    Note that data_filename is in the format: NUMBER_LABEL.data.jpg. This
    function takes such a string and returns the string for the test file.

    Params:
        data_filename - name of a data file

    Returns:
        The corresponding name of the test file for provided data file, as a
        string.
    """
    name = data_filename[5:].split(".data.jpg")[0]
    return "TESTS/" + name + ".test.png"

def generate_inputs():
    """
    Generates and populates the INPUTS/ directory with input pictures processed
    from the DATA/ directory, as well as the TESTS/ directory. The first half of
    the DATA/ files are used as INPUTS, and the second half is used for TESTS/
    """
    files = get_all_data_files()
    for f in files[ : len(files) / 2]:
        processed = edge_detect(f)
        cv2.imwrite(generate_input_name(f), processed)
    for f in files[len(files) / 2 : ]:
        processed = edge_detect(f)
        cv2.imwrite(generate_test_name(f), processed)

def get_input_producer():
    """
    Returns a TensorFlow String Input Producer using inputs from the INPUTS/
    directory.

    Params:
        None

    Returns:
        A Tensor
    """
    input_files = get_all_input_files()
    image_list = []
    for f in input_files:
        the_important_part_of_the_filename = f[7:].split(".input.png")[0]
        entry = the_important_part_of_the_filename.split("_")
        # entry[0] is the number
        # entry[1] is the label
        image_list.append(f + " " + entry[1])
    return tf.train.string_input_producer(image_list)

def get_test_producer():
    """
    Returns a TensorFlow String Input Producer using tests from the TESTS/
    directory.

    Params:
        None

    Returns:
        A Tensor
    """
    test_files = get_all_test_files()
    image_list = []
    for f in test_files:
        the_important_part_of_the_filename = f[7:].split(".test.png")[0]
        entry = the_important_part_of_the_filename.split("_")
        # entry[0] is the number
        # entry[1] is the label
        image_list.append(f + " " + entry[1])
    return tf.train.string_input_producer(image_list)

def get_train_set(filename_and_label_tensor):
    """
    Consumes a single filename and label as a ' '-delimited string.

    Params:
        filename_and_label_tensor: A scalar string tensor.

    Returns:
        Two tensors: the decoded image, and the string label.
    """
    filename, label = tf.decode_csv(filename_and_label_tensor, [[""], [""]], " ")
    file_contents = tf.read_file(filename)
    input_pic = tf.image.decode_png(file_contents)
    return input_pic, label

def get_test_set(filename_and_label_tensor):
    """
    NOTE: THIS FUNCTION IS SIMPLY A RENAME OF get_train_set() FOR CLARITY

    Consumes a single filename and label as a ' '-delimited string.

    Params:
        filename_and_label_tensor: A scalar string tensor.

    Returns:
        Two tensors: the decoded image, and the string label.
    """
    return get_train_set(filename_and_label_tensor)

