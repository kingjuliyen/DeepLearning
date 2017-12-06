

import json

class layer_defs:
    def __init__(self, json_data):
        self.json_root = json_data
        self.layer_params = { }
    
    def get_json_root(self):
        return self.json_root

    def get_layers_def_root(self):
        return self.json_root[layers_definition_str()]

    def get_num_layers(self):
        return (self.get_json_root())[num_layers_str()]

    def get_dim_0(self, l): # l should be layer number
        return self.get_layers_def_root()[str(l)] [dim_0_str()]

    def get_bias(self, l):
        return self.get_layers_def_root()[str(l)] [bias_str()]

    def needs_bias(self, l):
        return self.get_bias(l) == YES()
    
    def get_activation_type(self, l):
        return self.get_layers_def_root()[str(l)] [activation_str()]

    def add_layer_param(self, key, value):
        self.layer_params[key] = value

    def get_layer_param(self, key):
        return self.layer_params[key]

    def add_weight(self, l, weights):
        self.add_layer_param(weights_key(l), weights)
    def set_weight(self, l, weights):
        self.add_layer_param(weights_key(l), weights)
    def get_weight(self, l):
        return self.get_layer_param(weights_key(l))

    def add_bias(self, l, bias):
        self.add_layer_param(bias_key(l), bias)
    def set_bias(self, l, bias):
        self.add_layer_param(bias_key(l), bias)
    def get_bias(self, l):
        return self.get_layer_param(bias_key(l))

    def set_weight_dot_activation(self, l, W_dot_A):
        self.add_layer_param(W_dot_A_key(l), W_dot_A)

    def get_weight_dot_activation(self, l):
        return self.get_layer_param(W_dot_A_key(l))

    def set_activation(self, l, activation):
        self.add_layer_param(activation_key(l), activation)

    def get_activation(self, l):
        return self.get_layer_param(activation_key(l))

    def set_dZ(self, l, dZ):
        self.add_layer_param(dZ_key(l), dZ)
    def get_dZ(self, l):
        return self.get_layer_param(dZ_key(l))

    def set_dA(self, l, dA):
        self.add_layer_param(dA_key(l), dA)
    def get_dA(self, l):
        return self.get_layer_param(dA_key(l))

    def set_dW(self, l, dW):
        self.add_layer_param(dW_key(l), dW)
    def get_dW(self, l):
        return self.get_layer_param(dW_key(l))

    def set_dB(self, l, dB):
        self.add_layer_param(dB_key(l), dB)
    def get_dB(self, l):
        return self.get_layer_param(dB_key(l))

    def init_weights_and_biases(self, weights_setter, bias_setter):
        for l in range(1, self.get_num_layers()):
            weights = weights_setter(self.get_dim_0(l), self.get_dim_0(l-1)) #* 0.01
            bias = bias_setter(self.get_dim_0(l))
            # print("l == "+str(l)+ " weights.shape "+str(weights.shape)+"  "+str(bias.shape))
            self.add_weight(l, weights)
            self.add_bias(l, bias)
            # print(weights)

    def add_input_layer(self, train_x):
        self.set_activation(0, train_x)    
    def set_input_layer(self, X):
        self.set_activation(0, X)
    def get_input_layer(self):
        return self.get_activation(0)

    def print(self):
        for l in range(1, self.get_num_layers()):
            print("Layer "+ str(l))
            print("\t"+weights_key(l) + str(self.get_weight(l).shape))
            print("\t"+bias_key(l) + str(self.get_bias(l).shape))


# static defs follow
def layers_definition_str():
    return "layers_definition"

def dim_0_str():
    return "dim_0"

def num_layers_str():
    return "num_layers"

def bias_str():
    return "bias"

def weights_key(l):
    return weights_str() + layer_str(l)

def weights_str():
    return "weight"

def bias_key(l):
    return bias_str() + layer_str(l)

def W_dot_A_key(l):
    return W_dot_A_str() + layer_str(l)

def W_dot_A_str():
    return "W_dot_A"

def activation_key(l):
    return activation_str() + layer_str(l)

def activation_str():
    return "activation"

def dA_key(l):
    return dA_str() + layer_str(l)

def dA_str():
    return "dA"

def dZ_key(l):
    return dZ_str() + layer_str(l)

def dZ_str():
    return "dZ"

def dW_key(l):
    return dW_str() + layer_str(l)

def dW_str():
    return "dW"

def dB_key(l):
    return dB_str() + layer_str(l)

def dB_str():
    return "dB"

def layer_str(l):
    return str(l)

def YES():
    return "yes"