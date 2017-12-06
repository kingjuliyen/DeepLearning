

import sys
import numpy as np
import pandas as pd
import h5py

from image_utils import *

def npndarray_type():
    return str(type(np.zeros((0,0))))

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

def load_from_h5_infile(h5_file, ds_name = 'b83,2k'):
    h5f = h5py.File(h5_file, 'r')
    dataset = h5f[ds_name]
    npds = np.array(dataset)
    h5f.close()
    return npds # return numpy data set


class datastore:
    def __init__(self, name="_unnamed"):
        self.name = name

    def get_datastore(self, train_set, train_label, validation_percent, data_in_format):

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

        datastore[num_train_str()] = tr_end
        datastore[num_validation_str()] = vld_end - vld_begin
        return datastore

    def get_datastore_from_h5(self, train_h5, train_label_h5,
                            tr_xkey, tr_ykey, validation_percent):
        train = load_from_h5_infile(train_h5, tr_xkey)
        train_label = load_from_h5_infile(train_label_h5, tr_ykey)
        self.datastore = self.get_datastore(train, train_label, validation_percent, npndarray_type())

    def get_x_and_y(self, xgetter, ygetter, num_samples):
        _x = xgetter(self.datastore)
        if num_samples == -1:
            num_samples = int(_x.shape[0])

        x = _x.reshape(_x.shape[0], -1)
        x = normalize_pixel_value((x[0:num_samples ,:]).T)
        y = ygetter(self.datastore)
        y = y[0:num_samples]
        return x, y

    def get_train_x_and_y(self, num_train_samples):
        return self.get_x_and_y(get_train_set, get_train_set_label, num_train_samples)

    def print(self):
        print(' train set shape '+str(get_train_set(self.datastore).shape))
        print(' train set labels shape ' + str(get_train_set_label(self.datastore).shape))
        print(' validation set shape ' + str(get_validation_set(self.datastore).shape))
        print(' validation set labels shape ' + str(get_validation_set_label(self.datastore).shape))
        # print(' test set shape ' + str(get_test_set(self.datastore).shape))
        print(' number of training items ' + str(get_num_train_items(self.datastore)))
        print(' number of validation items ' + str(get_num_validation_items(self.datastore)))
        # print(' number of test items ' + str(get_num_test_items(self.datastore)))

    def get_holdout_set_x_and_y(self):
        hold_out_x = get_validation_set(self.datastore)
        hold_out_y = get_validation_set_label(self.datastore)

        hold_out_x = hold_out_x.reshape(hold_out_x.shape[0], -1)
        hold_out_x = normalize_pixel_value((hold_out_x[: ,:]).T)
        return hold_out_x, hold_out_y

