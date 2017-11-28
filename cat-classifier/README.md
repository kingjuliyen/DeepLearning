
Shallow single layer network cat-classifier example program

General info
=============
* numpy examples tried out on i7 mac
* Tensorflow examples tried and tested on p2.xlarge
* Currently only a hold out test is used can be extended for test set as well
* Large dataset is from kaggle cat classifier problem
* Before running the program set values in cat_or_not_params.py

Try with large dataset (max hold out set accuracy would be 57.3% approx)
=================================================================================
    Get dataset
    wget https://s3.us-east-2.amazonaws.com/dl-practice-set1/cat-or-not/tr002.h5
    wget https://s3.us-east-2.amazonaws.com/dl-practice-set1/cat-or-not/tr002.h5_label.h5

    Run with numpy backend
    python3 cat-or-not.numpy.py tr002.h5 tr002.h5_label.h5

    Run with tensorflow backend
    python3 cat-or-not.tensorflow.py tr002.h5 tr002.h5_label.h5

Try with small dataset (max hold out set accuracy would be 70.0% approx)
=================================================================================
    Get dataset
    wget https://s3.us-east-2.amazonaws.com/dl-practice-set1/cat-or-not/tr003.h5
    wget https://s3.us-east-2.amazonaws.com/dl-practice-set1/cat-or-not/te003.h5

    Run with numpy backend
    python3 cat-or-not.numpy.py tr003.h5 tr003.h5

    Run with tensorflow backend
    python3 cat-or-not.tensorflow.py tr003.h5 tr003.h5


source code layout
===================

├── README.md
├── cat-or-not.numpy.py              --> main script for numpy backend
├── cat-or-not.tensorflow.py         --> main script for numpy backend
├── cat_or_not_params.py             --> set hyper parameters and other settings before execution in this file
├── cat_or_not_utils.py              --> common functions of numpy and tensorflow backends

├── numpy_utils.py                   --> contains wrappers for numpy functions
├── tensorflow_utils.py              --> contains wrappers for tensorflow functions
└── workflow.py                      --> contains datastore datasets utility functions

├── image-utils                      --> utils of image resize conversions and conversion to h5 file
│   ├── gen-shell.py
│   ├── images-to-hd5.py
│   └── resize-img.py



How to copy code from local pc to linux cloud instance
========================================================

# bogus ip below replace with your cloud instance ip
export IP=123.124.125.126
rsync -Pav -e 'ssh -i mykey.pem'  \
     src_code_dir  \
     ubuntu@$IP:/home/ubuntu/dest_code_dir

