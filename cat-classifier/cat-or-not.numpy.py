

from cat_or_not_utils import *

def init_to_be_learnt_params(xshape):
    num_weights = int(xshape[0])
    weights = np.zeros((num_weights, 1))
    bias = 0.0
    return weights, bias

# Y known y labels for training set
# A activation vector calculated in do_props
def get_cost(Y, A, num_samples):
    v1 = Y * log(A)
    v2 =  (1.0 - Y) * log(1.0 - A)
    return (-1/num_samples) * sum(v1 + v2)

"""
do matrix multiplication weights.T X train_x, then apply sigmoid on it
"""
def do_props(train_x, train_y, weights, bias):
    num_samples = int(train_x.shape[1])
    # forward prop
    activation = sigmoid(np.dot(weights.T, train_x) + bias)
    cost = get_cost(train_y, activation, num_samples)
    # back_prop
    weights_grad = (1.0/num_samples) * np.dot(train_x, (activation - train_y).T)
    bias_grad = (1/num_samples) * sum(activation - train_y)

    cost = np.squeeze(cost)

    return weights_grad, bias_grad, cost


def watch_training_progress(cur_epoch, cost, weights, bias, ds):
    if cur_epoch % print_freq == 0:
        print("Epoch " + str(cur_epoch) + " cost " + str(cost))
        test_accuracy(ds, {"weights": weights, "bias": bias})
        print("\n")

"""=
train_x shape : ((num_pixel_rows * num_pixel_columns * num_channels), num_train_images) 
                num_channels will be 3 for RGB
                
weights shape : ((num_pixel_rows * num_pixel_columns * num_channels), 1)

bias shape : bias is a scalar

alpha - learning rate
num_epochs - how many train iterations

"""


# update grads => theta = theta - (theta_grad * alpha)
def train(ds, train_x, train_y, weights, bias, num_epochs, alpha, print_freq = 500):
    costs = []
    for cur_epoch in range(num_epochs):
        weights_grad, bias_grad, cost = do_props(train_x, train_y, weights, bias)

        weights -= (weights_grad * alpha)
        bias -= (bias_grad * alpha)

        watch_training_progress(cur_epoch, cost, weights, bias, ds)

    # plot_costs(costs, alpha)
    return weights, bias


def start_training(ds, num_train_samples = g_num_train_samples, num_epochs = g_num_epochs, alpha = g_alpha):
    train_x, train_y = get_train_x_and_y(ds, num_train_samples)
    weights, bias = init_to_be_learnt_params(train_x.shape)
    weights, bias = train(ds, train_x, train_y, weights, bias, num_epochs, alpha)
    return {"weights": weights, "bias": bias}


"""
python cat-or-not.py /tmp/tr002.h5 /tmp/tr002.h5_label.h5 /tmp/tr002.h5
"""

def main(argv):
    datastore = get_datastore_from_h5\
        (*check_and_parse_args(argv, 'cat-or-not.numpy.py'))
    print_datastore_details(datastore)
    learnt_params = start_training(datastore)
    test_accuracy(datastore, learnt_params)




#########################
# Main
#########################
if __name__== "__main__":
    sys.argv[0]
    main(sys.argv)

