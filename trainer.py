import img_parser
import tensorflow as tf
import numpy

def get_data_set():
    files = img_parser.get_all_data_files()
    return [img_parser.mat2array(img_parser.edge_detect(f)) for f in files]

def get_test_set():
    files = img_parser.get_all_test_files()
    return [img_parser.mat2array(img_parser.edge_detect(f)) for f in files]

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(input_layer, weight1, weight2, weight3, weight4, output_weight, p_keep_conv, p_keep_hidden):
    # Note: il_n denotes the nth intermediary layer
    il_1 = tf.nn.relu(tf.nn.conv2d(input_layer, weight1, [1, 1, 1, 1], 'SAME'))
    il_1 = tf.nn.max_pool(il_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    il_1 = tf.nn.dropout(il_1, p_keep_conv) # Apply dropout

    il_2 = tf.nn.relu(tf.nn.conv2d(il_1, weight2, [1, 1, 1, 1], 'SAME'))
    il_2 = tf.nn.max_pool(il_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    il_2 = tf.nn.dropout(il_2, p_keep_conv) # Apply dropout

    il_3 = tf.nn.relu(tf.nn.conv2d(il_2, weight3, [1, 1, 1, 1], 'SAME'))
    il_3 = tf.nn.max_pool(il_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    il_3 = tf.reshape(il_3, [-1, weight4.get_shape().as_list()[0]])
    il_3 = tf.nn.dropout(il_3, p_keep_conv)

    il_4 = tf.nn.relu(tf.matmul(il_3, weight4))
    il_4 = tf.nn.dropout(il_4, p_keep_hidden)

    output_layer = tf.matmul(il_4, output_weight)
    return output_layer

INPUT_PRODUCER = img_parser.get_input_producer()
train_img, train_label = img_parser.get_train_set(INPUT_PRODUCER.dequeue())

train_img.reshape(-1, 320, 240, 1)

X = tf.placeholder("float", [None, 320, 240, 1])

