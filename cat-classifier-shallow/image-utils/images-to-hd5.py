
import os
import os.path
import cv2
import numpy as np
import sys
import h5py
import random

def gen_h5(images_dir_name, outfile_name, gen_labels = False):
    hd5 = []
    labels = []

    shuffled_files_list = []
    for x in os.listdir(images_dir_name):
        shuffled_files_list.append(x)

    random.shuffle(shuffled_files_list)

    for file in shuffled_files_list:
        if not(file.endswith(".jpg")):
            continue


        if "cat" in file:
            labels.append(1)
        else:
            labels.append(0)

        #print(file)
        img = cv2.imread(images_dir_name+"/"+file)
        if(img is None):
            raise(file+" is not an image file")
        hd5.append(img)


    hd5 = np.array(hd5)
    hd5f = h5py.File(outfile_name, 'w')
    hd5f.create_dataset('images', data=hd5)
    hd5f.close()
    print("Finished writing " + outfile_name + " of shape "+ str(hd5.shape))

    labels = np.array(labels)
    labels_f_str = outfile_name+"_label.h5"
    labels_f = h5py.File(labels_f_str, 'w')
    labels_f.create_dataset('labels', data=labels)
    labels_f.close()
    print("Finished writing " + labels_f_str + " of shape "+ str(labels.shape))



def main(argv):
    if len(argv) < 3:
        raise('Usage images-to-hd5 images_dir hd5_out_filename.hd5 <gen_labels> ')
    _0, images_dir_name, outfile_name = argv
    gen_h5(images_dir_name, outfile_name)

if __name__== "__main__":
    sys.argv[0]
    main(sys.argv)
