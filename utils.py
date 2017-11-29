import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def crop_sub_imgs_fn2(x, is_random=True):
    n_img, m_img = x

    n_img, m_img = crop_multi([n_img, m_img], wrg=384, hrg=384, is_random=is_random)

    n_img = n_img / (255. / 2.)
    n_img = n_img - 1.

    m_img = m_img / 255.
    return n_img, m_img

def crop_sub_imgs_fn3(x, y, is_random=True) :
    result1 = []
    result2 = []


    return result1, result2

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_mask_fn(x):
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / 255.
    return x

def upsample_mask_fn(x):
    # We upsample mask images with upsampling factor r = 4.
    x = imresize(x, size=[384, 384], interp='bicubic', mode=None)
    x = x / 255.

    return x
    
def dilation_fn(x) :
    #x = x.astype(np.int)
    for i in range(3) :
        x[:,:,i] = dilation(x[:,:,i], 3)

    x = x / (255. / 2.)
    x = x - 1.    
    return x 