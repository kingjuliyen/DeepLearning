

import sys
import numpy as np
import pandas as pd
import h5py


def npndarray_type():
    return 'numpy.ndarray'

def train_set_str():
    return "train_set"

def train_set_label_str():
    return "train_set_label"

def validation_set_str():
    return "validation_set"

def validation_set_label_str():
    return "validation_set_label"

def test_set_str():
    return "test_set"

def num_train_str():
    return "num_train"

def num_validation_str():
    return "num_validation"

def num_test_str():
    return "num_test"

def get_train_set(datastore):
    return datastore[train_set_str()]

def get_train_set_label(datastore):
    return datastore[train_set_label_str()]

def get_validation_set(datastore):
    return datastore[validation_set_str()]

def get_validation_set_label(datastore):
    return datastore[validation_set_label_str()]

def get_test_set(datastore):
    return datastore[test_set_str()]

def get_num_train_items(datastore):
    return datastore[num_train_str()]

def get_num_validation_items(datastore):
    return datastore[num_validation_str()]

def get_num_test_items(datastore):
    return datastore[num_test_str()]

# str train_h5 - "/tmp/train.h5"
# str ds_name - dataset name "images"
# return npds 'numpy.ndarray'
def load_from_h5_infile(h5_file, ds_name = 'b83,2k'):
    h5f = h5py.File(h5_file, 'r')
    dataset = h5f[ds_name]
    npds = np.array(dataset)
    h5f.close()
    return npds # return numpy data set

def print_datastore_details(datastore):
    print(' train set shape '+str(get_train_set(datastore).shape))
    print(' train set labels shape ' + str(get_train_set_label(datastore).shape))
    print(' validation set shape ' + str(get_validation_set(datastore).shape))
    print(' validation set labels shape ' + str(get_validation_set_label(datastore).shape))
    print(' test set shape ' + str(get_test_set(datastore).shape))
    print(' number of training items' + str(get_num_train_items(datastore)))
    print(' number of validation items' + str(get_num_validation_items(datastore)))
    print(' number of test items' + str(get_num_test_items(datastore)))
    return

# train_set, test_set, should be of type 'numpy.ndarray'
# train_percent float
# function builds a data store dict
def get_datastore(train_set, train_label, test_set, validation_percent, data_in_format):
    assert(data_in_format is npndarray_type() )
    assert(len(train_set.shape) == 4)
    assert(len(test_set.shape) == 4)

    datastore = {}
    num_items, _0, _1, _2 = train_set.shape

    tr_begin = 0
    tr_end = int(num_items * float((100-validation_percent)/100))
    vld_begin = tr_end
    vld_end = num_items

    datastore[train_set_str()] = train_set[tr_begin:tr_end, :, :, :]
    datastore[train_set_label_str()] = train_label[tr_begin:tr_end]
    datastore[validation_set_str()] = train_set[vld_begin:vld_end, :, :, :]
    datastore[validation_set_label_str()] = train_label[vld_begin:vld_end]
    datastore[test_set_str()] = test_set

    datastore[num_train_str()] = tr_end
    datastore[num_validation_str()] = vld_end - vld_begin
    datastore[num_test_str()] = int(test_set.shape[0])

    return datastore

# str train_h5
# str test_h5
# returns datastore
def get_datastore_from_h5(train_h5, train_label_h5, test_h5,
                          tr_xkey, tr_ykey, te_xkey, te_ykey):
    train = load_from_h5_infile(train_h5, tr_xkey)
    train_label = load_from_h5_infile(train_label_h5, tr_ykey)
    test = load_from_h5_infile(test_h5, te_xkey)
    datastore = get_datastore(train, train_label, test, 20, npndarray_type())
    return datastore



