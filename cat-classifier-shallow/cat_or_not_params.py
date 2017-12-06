
use_large_set = 1

if use_large_set == 1:
    ################
    ## HYPER PARAMS
    ################

    g_num_epochs = 5000

    g_num_train_samples = 19000
    g_alpha = (0.005/2.0)


    g_num_validation_samples = 1000
    g_prediction_accuracy_threshold = 0.5


    ################

    print_freq = 1
    g_suppress_tensorflow_debug_msgs = 1
    g_run_test_set_accuracy_tests = 0

    # 'images', 'labels'
    # 'train_set_x', 'train_set_y', 'test_set_x', 'test_set_y'

    tr_xkey = 'images'
    tr_ykey = 'labels'
    te_xkey = 'images'
    te_ykey = 'labels'

    g_tr_xkey = tr_xkey
    g_tr_ykey = tr_ykey
    g_te_xkey = te_xkey
    g_te_ykey = te_ykey

else:
    ################
    ## HYPER PARAMS
    ################

    g_num_epochs = 100

    g_num_train_samples = 167
    g_alpha = (0.005/2.0)

    g_num_validation_samples = 40
    g_prediction_accuracy_threshold = 0.5


    ################


    print_freq = 1
    g_suppress_tensorflow_debug_msgs = 1
    g_run_test_set_accuracy_tests = 1

    # 'images', 'labels'
    # 'train_set_x', 'train_set_y', 'test_set_x', 'test_set_y'
    tr_xkey = 'train_set_x'
    tr_ykey = 'train_set_y'
    te_xkey = 'test_set_x'
    te_ykey = 'test_set_y'

    g_tr_xkey = tr_xkey
    g_tr_ykey = tr_ykey
    g_te_xkey = te_xkey
    g_te_ykey = te_ykey
