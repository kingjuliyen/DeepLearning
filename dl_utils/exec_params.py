

import json

class exec_params:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        print(" Reading dl params from "+self.json_file_path)

    def read_params(self):
        with open(self.json_file_path) as json_data:
            self.jsnd = json.load(json_data)

    def get_loaded_json_data(self):
        return self.jsnd

    def train_h5_file(self):
        return self.jsnd[train_h5_file_str()]
    def train_h5_label_file(self):
        return self.jsnd[train_h5_label_file_str()]
    def train_h5_x_key(self):
        return self.jsnd[train_h5_x_key_str()]
    def train_h5_y_key(self):
        return self.jsnd[train_h5_y_key_str()]

    def test_h5_file(self):
        return self.jsnd[test_h5_file_str()]
    def test_h5_label_file(self):
        return self.jsnd[test_h5_label_file_str()]
    def test_h5_x_key(self):
        return self.jsnd[test_h5_x_key_str()]
    def test_h5_y_key(self):
        return self.jsnd[test_h5_y_key_str()]

    def set_test_x(self, tx):
        self.test_x = tx
    def get_test_x(self):
        return self.test_x

    def set_test_y(self, ty):
        self.test_y = ty
    def get_test_y(self):
        return self.test_y

    def num_train_samples(self):
        return self.jsnd[num_train_samples_str()]
    def num_validation_samples(self):
        return self.jsnd[num_validation_samples_str()]
    def num_epochs(self):
        return self.jsnd[num_epochs_str()]
    def learning_rate(self):
        return self.jsnd[learning_rate_str()]
    def hold_out_set_percent(self): # % of training set
        return self.jsnd[hold_out_set_percent_str()]

    def run_hold_out_set_accuracy_tests(self):
        return self.jsnd[hold_out_set_accuracy_tests_str()] == YES()
    def run_train_set_accuracy_tests(self):
        return self.jsnd[train_set_accuracy_tests_str()] == YES()
    def run_test_set_accuracy_tests(self):
        return self.jsnd[test_set_accuracy_tests_str()] == YES()

    def set_dot_product(self, dot_product):
        self.dot_product = dot_product
    def get_dot_product(self):
        return self.dot_product
    
    def set_cost_function(self, cost_function):
        self.cost_function = cost_function
    def get_cost_function(self):
        return self.cost_function
    
    def set_relu(self, relu):
        self.relu = relu
    def get_relu(self):
        return self.relu

    def set_relu_grad(self, relu_grad):
        self.relu_grad = relu_grad
    def get_relu_grad(self):
        return self.relu_grad

    def set_sigmoid(self, sigmoid):
        self.sigmoid = sigmoid
    def get_sigmoid(self):
        return self.sigmoid

    def set_sigmoid_grad(self, sigmoid_grad):
        self.sigmoid_grad = sigmoid_grad
    def get_sigmoid_grad(self):
        return self.sigmoid_grad

    def get_activation_function_grad(self, activation_type):
        if activation_type == relu_str():
            return self.get_relu_grad()
        elif activation_type == sigmoid_str():
            return self.get_sigmoid_grad()

    def set_cost_grad(self, bootstrap_grad):
        self.cost_grad = bootstrap_grad
    def get_cost_grad(self):
        return self.cost_grad

    def set_sum_function(self, sum_function):
        self.sum_function = sum_function
    def get_sum_function(self):
        return self.sum_function

    def get_activation_function(self, activation_type):
        if activation_type == relu_str():
            return self.get_relu()
        elif activation_type == sigmoid_str():
            return self.get_sigmoid()

# end class exec_params:

##
# Some static helpers
##
def train_h5_file_str():
    return "train_h5_file"
def train_h5_label_file_str():
    return "train_h5_label_file"
def train_h5_x_key_str():
    return "train_h5_x_key"   
def train_h5_y_key_str():
    return "train_h5_y_key"

def test_h5_file_str():
    return "test_h5_file"
def test_h5_label_file_str():
    return "test_h5_label_file"
def test_h5_x_key_str():
    return "test_h5_x_key"
def test_h5_y_key_str():
    return "test_h5_y_key"


def num_train_samples_str():
    return "num_train_samples"

def num_validation_samples_str():
    return "num_validation_samples"

def num_epochs_str():
    return "num_epochs"

def relu_str():
    return "relu"

def sigmoid_str():
    return "sigmoid"

def learning_rate_str():
    return "learning_rate"

def hold_out_set_percent_str():
    return "hold_out_set_percent"

def hold_out_set_accuracy_tests_str():
    return "hold_out_set_accuracy_tests"
def train_set_accuracy_tests_str():
    return "train_set_accuracy_tests"
def test_set_accuracy_tests_str():
    return "test_set_accuracy_tests"

def YES():
    return "yes"
