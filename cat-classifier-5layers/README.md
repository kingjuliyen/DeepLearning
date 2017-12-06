
Five layer network cat-classifier example program

General info
=============
* numpy examples tried out on i7 mac
* Before running the program set values in 5_layers.json
* To run the script do
    python3 cat_or_not_5_layers_numpy.py 5_layers.json

Try with small dataset (max test  set accuracy would be 77.55% approx)
======================================================================
    Should see something like this after the end of program

 	trainset results
 	accuracy_percentage 95.80838323353294%  correct_prediction 160

 	holdout set results
 	accuracy_percentage 100.0%  correct_prediction 3

 	testset results
 	accuracy_percentage 77.55102040816327%  correct_prediction 38

Get dataset
===========
    wget https://s3.us-east-2.amazonaws.com/dl-practice-set1/cat-or-not/tr003.h5
    wget https://s3.us-east-2.amazonaws.com/dl-practice-set1/cat-or-not/te003.h5


source code layout
===================

├── README.md
├── cat_or_not_5_layers_numpy.py              --> main script for numpy backend
├── 5_layers.json                             --> contains dl input parameters in json format

├── ../dl_utils                           --> general utils dir for deep learning
│       ├── exec_params.py                    --> contains dl parameters key store
│       ├── layer_defs.py                     --> contains dl layer book keeping info
│       ├── datastore_utils.py                --> contains h5 utils code
│       ├── vanilla_optimizer.py              --> simple gradient descent optimizer
│       └── image_utils.py                    --> contains pixel normalizer function for now
