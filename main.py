#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import math 

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
from scipy.misc import imresize

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
lr_text_init = config.TRAIN.lr_text_init

## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def psnr(img1, img2) :
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    train_hr_mask_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_mask_img_path, regx='.*.png', printable=False))
    train_hr_natural_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_natural_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_hr_mask_imgs = read_all_imgs(train_hr_mask_img_list, path=config.TRAIN.hr_mask_img_path, n_threads=32)
    train_hr_natural_imgs = read_all_imgs(train_hr_natural_img_list, path=config.TRAIN.hr_natural_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    t_mask_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_mask_image_input_to_SRGAN_generator')
    t_mask_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name = 't_mask_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_natural_d, logits_natural_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_natural_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_text_d, logits_text_real = SRGAN_d2(tf.multiply(t_mask_image, t_target_image), is_train=True, reuse=False)
    _,     logits_text_fake = SRGAN_d2(tf.multiply(t_mask_image, net_g.outputs), is_train=True, reuse=True)
    
    net_g.print_params(False)
    net_natural_d.print_params(False)
    net_text_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg


    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    t_mask_image_224 = tf.image.resize_images(t_mask_image, size=[224, 224], method=0, align_corners=False) 

    _, vgg_mask_predict_emb = Vgg19_simple_api((tf.multiply(t_mask_image_224, t_target_image_224)+1)/2, reuse=True)
    _, vgg_mask_target_emb = Vgg19_simple_api((tf.multiply(t_mask_image_224, t_predict_image_224)+1)/2, reuse=True)


    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_natural_loss1 = tl.cost.sigmoid_cross_entropy(logits_natural_real, tf.ones_like(logits_natural_real), name='d11')
    d_natural_loss2 = tl.cost.sigmoid_cross_entropy(logits_natural_fake, tf.zeros_like(logits_natural_fake), name='d12')
    d_natural_loss = d_natural_loss1 + d_natural_loss2

    d_text_loss1 = tl.cost.sigmoid_cross_entropy(logits_text_real, tf.ones_like(logits_text_real), name='d21')
    d_text_loss2 = tl.cost.sigmoid_cross_entropy(logits_text_fake, tf.zeros_like(logits_text_fake), name='d22')
    d_text_loss = d_text_loss1 + d_text_loss2

    g_gan_natural_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_natural_fake, tf.ones_like(logits_natural_fake), name='g1')
    g_gan_text_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_text_fake, tf.ones_like(logits_text_fake), name='g2')

    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_natural_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    vgg_text_loss = 2e-6 * tl.cost.mean_squared_error(vgg_mask_predict_emb.outputs, vgg_mask_target_emb.outputs, is_mean=True)


    # loss compete
    g_natural_loss = g_gan_natural_loss + vgg_natural_loss
    g_text_loss = g_gan_text_loss + vgg_text_loss


    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_natural_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)
    d_text_vars = tl.layers.get_variables_with_name('SRGAN_d2', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
        lr_text_v = tf.Variable(lr_text_init, trainable=False)

    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_natural_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_natural_loss, var_list=g_vars)
    d_natural_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_natural_loss, var_list=d_natural_vars)

    g_text_optim = tf.train.AdamOptimizer(lr_text_v, beta1=beta1).minimize(g_text_loss, var_list=g_vars)
    d_text_optim = tf.train.AdamOptimizer(lr_text_v, beta1=beta1).minimize(d_text_loss, var_list=d_text_vars)


    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_natural_{}.npz'.format(tl.global_flag['mode']), network=net_natural_d)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_text_{}.npz'.format(tl.global_flag['mode']), network=net_text_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npy from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit+'/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan+'/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))

    if os.path.exists(checkpoint_dir + '/g_srgan_init.npz') is False :
        print(" ** fixed learning rate: %f (for init G)" % lr_init)
        for epoch in range(0, n_epoch_init+1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0

            ## If your machine cannot load all images into memory, you should use
            ## this one to load batch of images while training.
            # random.shuffle(train_hr_img_list)
            # for idx in range(0, len(train_hr_img_list), batch_size):
            #     step_time = time.time()
            #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            ## If your machine have enough memory, please pre-load the whole train set.
            for idx in range(0, len(train_hr_imgs), batch_size):
                step_time = time.time()

                b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
                b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
                ## update G
                errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
            print(log)

            ## quick evaluation on train set
            if (epoch != 0) and (epoch % 10 == 0):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

            ## save model
            if (epoch != 0) and (epoch % 10 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)


    test_dir_gan = "samples/{}_test".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(test_dir_gan)
    isFirst = True
    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            sess.run(tf.assign(lr_text_v, lr_text_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            sess.run(tf.assign(lr_text_v, lr_text_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()

            b_imgs = train_hr_imgs[idx : idx + batch_size]
            b_mask_imgs = train_hr_mask_imgs[idx : idx + batch_size]

            # b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)

            b_data = tl.prepro.threading_data([_ for _ in zip(b_imgs, b_mask_imgs)], fn=crop_sub_imgs_fn2, is_random=True)
            b_imgs_384, b_mask_imgs_384 = b_data.transpose((1,0,2,3,4))

            if isFirst is True :
                tl.vis.save_images(b_imgs_384, [ni, ni], test_dir_gan+'/img.png')
                tl.vis.save_images(b_mask_imgs_384, [ni, ni], test_dir_gan+'/mask.png')
                isFirst = False

            b_mask_bicubic_imgs_384 = tl.prepro.threading_data(b_mask_imgs_384, fn=dilation_fn)
            #b_imgs_384, b_mask_imgs_384 = crop_sub_imgs_fn3(b_imgs, b_mask_imgs, True)

            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            '''
            b_mask_imgs_96 = tl.prepro.threading_data(b_mask_imgs_384, fn=downsample_mask_fn)
            b_mask_bicubic_imgs_384 = tl.prepro.threading_data(b_mask_imgs_96, fn=upsample_mask_fn)
            
            b_mask_bicubic_imgs_384[b_mask_bicubic_imgs_384 > 0] = 1.
            '''
            b_natural_imgs = train_hr_natural_imgs[idx : idx + batch_size]
            b_natural_imgs_384 = tl.prepro.threading_data(b_natural_imgs, fn=crop_sub_imgs_fn, is_random=True)
            b_natural_imgs_96 = tl.prepro.threading_data(b_natural_imgs_384, fn=downsample_fn)

            #####

            ## update D
            errD, _ = sess.run([d_natural_loss, d_natural_optim], {t_image: b_natural_imgs_96, t_target_image: b_natural_imgs_384})
            errD2, _ = sess.run([d_text_loss, d_text_optim], {t_image : b_imgs_96, t_mask_image: b_mask_bicubic_imgs_384, t_target_image: b_imgs_384})

            ## update G
            errG, errA, _ = sess.run([g_text_loss, g_gan_text_loss, g_text_optim], { t_image : b_imgs_96, t_target_image : b_imgs_384, t_mask_image: b_mask_bicubic_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_natural_loss: %.8f g_loss: %.8f (adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errA))

            #print("Epoch [%2d/%2d] %4d time: %4.4fs, d_natural_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            

            errG2, errA2, _ = sess.run([g_natural_loss, g_gan_natural_loss, g_natural_optim], {t_image: b_natural_imgs_96, t_target_image: b_natural_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_natural_loss: %.8f g_loss: %.8f (adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD2, errG2, errA2))

            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_natural_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_natural_d.all_params, name=checkpoint_dir+'/d_natural_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_text_d.all_params, name=checkpoint_dir+'/d_text_{}.npz'.format(tl.global_flag['mode']), sess=sess)

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    load_params = tl.files.load_npz(checkpoint_dir, '/g_srgan.npz')

    num_of_images = len(valid_lr_imgs)
    #num_of_images = 1

    for idx in range(0, num_of_images) :

        ###========================== DEFINE MODEL ============================###
        #imid = 64 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        #imid = 0
        valid_lr_img = valid_lr_imgs[idx]
        valid_hr_img = valid_hr_imgs[idx]
            # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        #valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]

        valid_lr_img = imresize(valid_lr_img, 0.25, interp='bicubic', mode=None)
        valid_lr_img = (valid_lr_img / 127.5) - 1
        valid_hr_img = (valid_hr_img / 127.5) - 1

        # print(valid_lr_img.min(), valid_lr_img.max())

        size = valid_lr_img.shape
        t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
        # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

        with tl.ops.suppress_stdout():
            if idx == 0 :
                net_g = SRGAN_g(t_image, is_train=False, reuse=False)
            else :
                net_g = SRGAN_g(t_image, is_train=False, reuse=True)


        ###========================== RESTORE G =============================###
        #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        #tl.layers.initialize_global_variables(sess)
        #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

        if idx == 0 :
            tl.files.assign_params(sess, load_params, net_g)

        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))

        #print("Image %d : LR size: %s /  generated HR size: %s" % (idx, size, out.shape))
        #print("Image %d : LR size: %s /  generated HR size: %s / PSNR : %f " % (idx, size, out.shape, psnr(out[0], valid_hr_img))) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        # print("[*] save images")

        tl.vis.save_image(out[0], save_dir + '/' + str(idx) + '_valid_gen.png')
        tl.vis.save_image(valid_lr_img, save_dir + '/' + str(idx) +'_valid_lr.png')
        tl.vis.save_image(valid_hr_img, save_dir + '/' + str(idx) +'_valid_hr.png')

        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, save_dir + '/' + str(idx) + '_valid_bicubic.png')

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
