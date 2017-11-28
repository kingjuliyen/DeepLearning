

import os
import tensorflow as tf


def to_tensor_f32(v):
    return tf.convert_to_tensor(v, tf.float32)

def tensor_matmul(a, b, is_a_transpose, is_b_transpose):
    return tf.matmul(a, b, transpose_a = is_a_transpose, transpose_b = is_b_transpose)

def tensor_sigmoid(z):
    return tf.sigmoid(z)

def tensor_log(k):
    return tf.log(k)

def tensor_sum(a):
    return tf.reduce_sum(a)

def suppress_tensorflow_debug_msgs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

