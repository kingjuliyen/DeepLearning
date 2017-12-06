
from tensorflow_utils import *
from cat_or_not_utils import *

def init_to_be_learnt_params_tf(xshape):
    num_weights = int(xshape[0])
    weights = tf.zeros([num_weights, 1], tf.float32)
    bias = to_tensor_f32(0.0)
    return weights, bias

def get_cost_tf(Y, A, num_samples):
    v1 = Y * tensor_log(A)
    v2 = (1.0 - Y) * tensor_log(1.0 - A)
    v3 = tensor_sum(v1 + v2)
    v4 = (-1/num_samples) * v3
    return v4

def do_props_tf(train_x, train_y, weights, bias):
    num_samples = int(train_x.shape[1])

    # forward prop
    activation = tensor_sigmoid(tensor_matmul(weights, train_x, True, False))
    cost = get_cost_tf(train_y, activation, num_samples)

    weights_grad = (1.0 / num_samples) * tensor_matmul(train_x, (activation - train_y), False, True)
    bias_grad = (1/num_samples) * tensor_sum(activation - train_y)

    with tf.Session() as sess:
        weights_grad, bias_grad, cost = sess.run([weights_grad, bias_grad, cost])

    return weights_grad, bias_grad, cost


def watch_training_progress_tf(cur_epoch, cost, weights, bias, ds):
    if cur_epoch % print_freq == 0:
        print("Epoch " + str(cur_epoch) + " cost " + str(cost))

        with tf.Session() as sess:
            _weights, _bias = sess.run([weights, bias])
            test_accuracy(ds, {"weights": _weights, "bias": _bias})
        print("\n")

def train_tf(ds, train_x, train_y, weights, bias, num_epochs, alpha):
    costs = []
    cost = 0

    for cur_epoch in range(num_epochs):
        weights_grad, bias_grad, cost = do_props_tf(train_x, train_y, weights, bias)
        weights -= (weights_grad * alpha)
        bias -= (bias_grad * alpha)
        costs.append(cost)
        watch_training_progress_tf(cur_epoch, cost, weights, bias, ds)

    with tf.Session() as sess:
        weights, bias = sess.run([weights, bias])

    return weights, bias


def start_training_tf(ds, num_train_samples = g_num_train_samples, num_epochs = g_num_epochs, alpha = g_alpha):
    train_x, train_y = get_train_x_and_y(ds, num_train_samples)
    train_x = to_tensor_f32(train_x)
    train_y = to_tensor_f32(train_y)

    weights, bias = init_to_be_learnt_params_tf(train_x.shape)
    weights, bias = train_tf(ds, train_x, train_y, weights, bias, num_epochs, alpha)
    return 0


"""
python cat-or-not.py /tmp/tr002.h5 /tmp/tr002.h5_label.h5 /tmp/tr002.h5
"""

def main(argv):
    if g_suppress_tensorflow_debug_msgs == 1:
        suppress_tensorflow_debug_msgs()

    datastore = get_datastore_from_h5(*check_and_parse_args(argv, 'cat-or-not.tensorflow.py'))
    # print_datastore_details(datastore)
    learnt_params = start_training_tf(datastore)
    # test_accuracy(datastore, learnt_params)

#########################
# Main
#########################
if __name__== "__main__":
    sys.argv[0]
    main(sys.argv)
