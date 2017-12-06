
# cat_or_not_gt1L_main_numpy:
#       cat or not classifier with greater than 1 layer using numpy backend

# PYTHONPATH=../dl_utils python3 cat_or_not_gt1L_main_numpy.py ep1.json

import sys
sys.path.append('../dl_utils')

import numpy as np
import math

from layer_defs import *
from exec_params import *
from datastore_utils import *
from vanilla_optimizer import *

np.random.seed(1)
np.set_printoptions(suppress=True)

def weights_setter(dim_0, dim_1):
    return np.random.randn(dim_0, dim_1) / np.sqrt(dim_1)

def bias_setter(dim_0):
    return np.zeros((dim_0, 1))

def relu(v):
    return np.maximum(v, 0)

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def relu_grad(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def sigmoid_grad(dA, Z):
    g_dash = dA * sigmoid(Z) * (1 - sigmoid(Z))
    return g_dash

def cost_grad(Y, last_activation):
    return -(np.divide(Y, last_activation) - np.divide(1 - Y, 1 - last_activation))

def cost_function(Yhat, Y):
    m = Y.shape[0]
    cost = (1./m) * (-np.dot(Y,np.log(Yhat).T) - np.dot(1-Y, np.log(1-Yhat).T))
    cost = np.squeeze(cost)
    return cost

def sum_function(v):
    return np.sum(v, axis = 1, keepdims = True)

def set_callbacks(ep):
    ep.set_dot_product(np.dot)
    ep.set_relu(relu)
    ep.set_sigmoid(sigmoid)
    ep.set_cost_grad(cost_grad)
    ep.set_cost_function(cost_function)
    ep.set_sum_function(sum_function)
    ep.set_relu_grad(relu_grad)
    ep.set_sigmoid_grad(sigmoid_grad)

def load_test_files(ep):
    print("\n load_test_files")
    ds = datastore("cat-or-dog")
    ds.get_datastore_from_h5(ep.test_h5_file(), ep.test_h5_label_file(),
                                ep.test_h5_x_key(), ep.test_h5_y_key(), 1)
    test_x, test_y = ds.get_train_x_and_y(-1)
    print(" test set shape "+str(test_x.shape))
    print(" test set label shape "+str(test_y.shape))

    ep.set_test_x(test_x)
    ep.set_test_y(test_y)

def start_training(ep):
    ds = datastore("cat-or-dog")
    ds.get_datastore_from_h5(ep.train_h5_file(), ep.train_h5_label_file(),
                                ep.train_h5_x_key(), ep.train_h5_y_key(), ep.hold_out_set_percent())
    ds.print()
    train_x, train_y = ds.get_train_x_and_y(ep.num_train_samples())
    hold_out_x, hold_out_y = ds.get_holdout_set_x_and_y()
    if ep.run_test_set_accuracy_tests():
        load_test_files(ep)

    layer_def = layer_defs(ep.get_loaded_json_data())
    layer_def.init_weights_and_biases(weights_setter, bias_setter)
    layer_def.add_input_layer(train_x)

    set_callbacks(ep)

    optimizer = vanilla_optimizer(layer_def, train_x, train_y, ep, hold_out_x, hold_out_y )
    optimizer.train()

def start_classifier(json_file):
    params = exec_params(json_file)
    params.read_params()
    start_training(params)

## PYTHONPATH=../dl_utils python3 cat_or_not_gt1L_main_numpy.py ep1.json
def main(argv):
    num_args = len(argv)
    if num_args < 2:
        print("Correct Usage\n\tpython3 cat_or_not_gt1L_main_numpy exec_params1234.json")
        return
    in_json_file = argv[1]
    start_classifier(in_json_file)

#########################
# Main
#########################
if __name__== "__main__":
    main(sys.argv)

