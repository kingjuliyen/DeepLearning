


from numpy_utils import *
from workflow import *
from cat_or_not_params import *

def normalize_pixel_value(v):
    return v/255.0

def get_x_and_y(ds, xgetter, ygetter, num_samples):
    _x = xgetter(ds)
    x = _x.reshape(_x.shape[0], -1)
    x = normalize_pixel_value((x[0:num_samples ,:]).T)
    y = ygetter(ds)
    y = y[0:num_samples]
    return x, y

def get_train_x_and_y(ds, num_train_samples):
    return get_x_and_y(ds, get_train_set, get_train_set_label, num_train_samples)

def get_validation_x_and_y(ds, num_validation_samples):
    return get_x_and_y(ds, get_validation_set, get_validation_set_label, num_validation_samples)


def test_accuracy(ds, learnt_params, num_validation_samples = g_num_validation_samples, \
                  threshold = g_prediction_accuracy_threshold):
    if g_run_test_set_accuracy_tests != 1 :
        return

    weights = learnt_params["weights"]
    bias = learnt_params["bias"]
    validation_x, validation_y = get_validation_x_and_y(ds, num_validation_samples)
    predictions = np.zeros((1, num_validation_samples))
    p = sigmoid(np.dot(weights.T, validation_x))
    p[p > threshold] = 1
    p[p <= threshold] = 0
    ac = p == validation_y
    correct_prediction = int((ac[ac==True]).shape[0])
    accuracy_percentage = correct_prediction / num_validation_samples * 100.0
    print("test accuracy_percentage "+str(accuracy_percentage)+"% " + \
          " correct_prediction " + str(correct_prediction))

# 3rd arg test_h5 unused for now
def check_and_parse_args(argv, script_name = 'cat-or-not'):
    if len(argv) < 3:
        print ('\n InCorrect Usage error, needs: \n\t' + script_name + ' train.h5 label_train.h5 \n')
        raise('Usage ' + script_name +' train.h5 label_train.h5 test.h5')
    train_h5 = argv[1]
    train_label_h5 = argv[2]
    test_h5 =  argv[1] # bogus and needs to be changed when test set validation is implemented
    return(train_h5, train_label_h5, test_h5, g_tr_xkey, g_tr_ykey, g_te_xkey, g_te_ykey)

"""
def plot_costs(costs, alpha):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per thousands)')
    plt.title("Learning rate " + str(alpha))
    plt.show()
    return
"""
