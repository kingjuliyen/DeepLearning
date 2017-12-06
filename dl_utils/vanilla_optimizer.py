import sys
import numpy as np

from layer_defs import *
from exec_params import *

class vanilla_optimizer:
    def __init__(self, layer_def, train_x, train_y, dl_params,  hold_out_x, hold_out_y ):
        self.layer_def = layer_def
        self.train_x = train_x
        self.train_y = train_y
        self.hold_out_x = hold_out_x
        self.hold_out_y = hold_out_y
        self.num_samples = int(self.train_y.shape[0])
        self.dl_params = dl_params
        self.dot_product = self.dl_params.get_dot_product()
        self.cost_function = self.dl_params.get_cost_function()
        self.get_cost_grad = self.dl_params.get_cost_grad()
        self.sum_function = self.dl_params.get_sum_function()

    def predict(self, hx):
        ldf = self.layer_def
        train_x = ldf.get_input_layer()
        ldf.set_input_layer(hx)
        activation = ldf.get_input_layer()
        for l in range(1, ldf.get_num_layers()):
            weight = ldf.get_weight(l)
            bias = ldf.get_bias(l)
            w_dot_a = self.dot_product(weight, activation) + bias
            activation_function = self.get_activation_function(l)
            activation = activation_function(w_dot_a)
        ldf.set_input_layer(train_x)
        return activation

    def run_test(self, threshold, _x, _y):
        ldf = self.layer_def
        p = self.predict(_x)
        p[p > threshold] = 1
        p[p <= threshold] = 0
        ac = p == _y
        correct_prediction = int((ac[ac==True]).shape[0])
        num_validation_samples = float(_y.shape[0])
        accuracy_percentage = correct_prediction / num_validation_samples * 100.0
        print(" \taccuracy_percentage "+str(accuracy_percentage)+"% " + \
            " correct_prediction " + str(correct_prediction))

    def run_holdout_tests(self, threshold = 0.5):
        print("\n \tholdout set results")
        _x = self.hold_out_x
        _y = self.hold_out_y
        self.run_test(threshold, _x, _y)

    def run_trainset_tests(self, threshold = 0.5):
        print("\n \ttrainset results")
        _x = self.train_x
        _y = self.train_y
        self.run_test(threshold, _x, _y)

    def run_testset_prediction(self, threshold = 0.5):
        print("\n \ttestset results")
        _x = self.dl_params.get_test_x()
        _y = self.dl_params.get_test_y()
        self.run_test(threshold, _x, _y)

    def get_activation_function(self, l):
        activation_type = self.layer_def.get_activation_type(l)
        return self.dl_params.get_activation_function(activation_type)

    def do_forward_propagation(self, layer_def):
        for l in range(1, layer_def.get_num_layers()):
            weight = layer_def.get_weight(l)
            bias = layer_def.get_bias(l)
            activation = layer_def.get_activation(l-1)
            w_dot_a = self.dot_product(weight, activation) + bias
            activation_function = self.get_activation_function(l)
            activation = activation_function(w_dot_a)
            layer_def.set_weight_dot_activation(l, w_dot_a)
            layer_def.set_activation(l, activation)

    def compute_cost_grad(self):
        ldf = self.layer_def
        Y = self.train_y
        last_activation = ldf.get_activation(ldf.get_num_layers()-1)
        cg = self.get_cost_grad(Y, last_activation)
        ldf.set_dA(ldf.get_num_layers()-1, cg)
    
    def get_g_dash_Z(self, l, dA):
        activation_type = self.layer_def.get_activation_type(l)
        ag = self.dl_params.get_activation_function_grad(activation_type)
        Z = self.layer_def.get_weight_dot_activation(l)
        r = ag(dA, Z)
        return r

    def compute_layerwise_grad(self, l):
        ldf = self.layer_def

        W = ldf.get_weight(l)
        Aprev = ldf.get_activation(l-1)
        dA = ldf.get_dA(l)
        dZ = self.get_g_dash_Z(l, dA)                              # dZ(l) = dA[l] ∗ g′(Z[l])
        dAprev = self.dot_product(W.T, dZ)                         # dA(l-1) = W(l).T . dz(l)
        dW = (1/self.num_samples) * self.dot_product(dZ, Aprev.T)  # dW(l) = (1/m) * dz . A(l-1).T
        dB = (1/self.num_samples) * self.sum_function(dZ)          # dB(l) = (1/m) * sum(dz(l))

        ldf.set_dA(l-1, dAprev)
        ldf.set_dZ(l, dZ)
        ldf.set_dW(l, dW)
        ldf.set_dB(l, dB)

    def update_grads(self):
        ldf = self.layer_def
        alpha = self.dl_params.learning_rate()
        layers = range(1, ldf.get_num_layers())
        for l in layers:
            W = ldf.get_weight(l)
            B = ldf.get_bias(l)
            W = W - alpha * ldf.get_dW(l)
            B = B - alpha * ldf.get_dB(l)
            ldf.set_weight(l, W)
            ldf.set_bias(l, B)

    def do_backward_propagation(self):
        layer_def = self.layer_def
        self.compute_cost_grad()

        layers = range(1, layer_def.get_num_layers())
        for l in reversed(layers):
            self.compute_layerwise_grad(l)
        self.update_grads()
   
    def get_cost(self):
        layer_def = self.layer_def
        Y = self.train_y
        Yhat = layer_def.get_activation(layer_def.get_num_layers()-1)
        return self.cost_function(Yhat, Y)

    def train(self):
        print("\n\n optimizer training started")
        num_epochs = self.dl_params.num_epochs()
        layer_def = self.layer_def

        for epoch in range(1, num_epochs+1):
            self.cur_epoch = epoch
            self.do_forward_propagation(layer_def)
            cost = self.get_cost()

            self.do_backward_propagation()
            if(epoch % 500 ) == 0:
                print("\n\n\n epoch "+str(epoch) + " cost "+str(cost))
                if self.dl_params.run_train_set_accuracy_tests():
                    self.run_trainset_tests()
                if self.dl_params.run_hold_out_set_accuracy_tests():
                    self.run_holdout_tests()
                if self.dl_params.run_test_set_accuracy_tests():
                    self.run_testset_prediction()

        print("\n\n\n")
        if self.dl_params.run_train_set_accuracy_tests():
            self.run_trainset_tests()
        if self.dl_params.run_hold_out_set_accuracy_tests():
            self.run_holdout_tests()
        if self.dl_params.run_test_set_accuracy_tests():
            self.run_testset_prediction()
