


import os.path
import cv2
import numpy as np
import sys

def get_centred(w, h, pad_sz = 3):
    sz = sz = int(max(h, w) + pad_sz)
    hs = int((sz/2) - (h/2)) # hs = height start
    he = int(hs + h)         # he = height end
    ws = int((sz/2) - (w/2)) # ws = width start
    we = int(ws + w)         # we = width end
    return(hs, he, ws, we, sz)

def get_squared_image(orig_img):
    h, w, nc = orig_img.shape
    hs, he, ws, we, sz = get_centred(w, h)
    squared_image = np.zeros((sz, sz, 3), np.uint8)
    squared_image[hs: he, ws: we] = orig_img
    return squared_image

def resize_image(infile, to_height, to_width):
    orig_image = cv2.imread(infile)
    h, w, nc = orig_image.shape
    if nc != 3:
        raise Exception('fn resize_image nc != 3')
    print("Original image height "+str(h)+ " width " + str(w) + " num_channel "+str(nc))

    squared_image = get_squared_image(orig_image)
    resized_image = cv2.resize(squared_image, (to_height, to_width), interpolation=cv2.INTER_AREA)
    return resized_image

def write_out_file(outdir, orig_file_name, img, to_height, to_width):
    out_file_name = outdir+"/"+str(to_height) +"X"+str(to_width)+"."+os.path.basename(orig_file_name)
    cv2.imwrite(out_file_name, img)
    print(" Writing resized image to "+out_file_name)

def main(argv):
    if len(argv) < 5:
        raise Exception('Usage e.g python resize-img 64 64 /tmp')
    _0, infile, to_height, to_width, out_dir = argv
    to_height = int(to_height)
    to_width = int(to_width)
    print("## "+\
          " Converting "+infile+" to height "+str(to_height)+ " to width "+str(to_width) \
          +" ## ")
    resized_image = resize_image(infile, to_height, to_width)
    write_out_file(out_dir, infile, resized_image, to_height, to_width)

if __name__== "__main__":
    sys.argv[0]
    main(sys.argv)

