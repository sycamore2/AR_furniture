import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import  layers
import numpy as np
import math
import scipy
from scipy.signal import convolve2d

block_size = 32
def parse_function(example_proto):
    features = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(example_proto, features)
    data = tf.io.decode_raw(parsed_features['data'], tf.float32)
    label = tf.io.decode_raw(parsed_features['label'], tf.float32)
    data = tf.reshape(data, (160, 160, 4))
    label = tf.reshape(label, (160, 160, 4))

    return data, label

#######################################################################

def data_example(binary_data,binary_label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'data': bytes_feature(binary_data),
        'label': bytes_feature(binary_label)
    }))
    return example

#######################################################################

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#######################################################################

def _psnr(image,image_rec):
    image = image.numpy()
    image_rec = image_rec.numpy()
    diff = image - image_rec
    diff = diff.flatten('C')
    # diff = diff.eval()
    # a, b = diff.shape
    # diff = diff.reshape((a*b,1))
    rmse = math.sqrt(np.mean(diff ** 2.))
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr

#######################################################################

def psnr_4d(image_3d,image_rec_3d):

    _psnr_key_1 = _psnr(image_3d[:, :, 0],image_rec_3d[:, :, 0])
    _psnr_cs_1 = _psnr(image_3d[:, :, 1],image_rec_3d[:, :, 1])
    _psnr_cs_2 = _psnr(image_3d[:, :, 2], image_rec_3d[:, :, 2])
    _psnr_cs_3 = _psnr(image_3d[:, :, 3], image_rec_3d[:, :, 3])

    return _psnr_key_1,_psnr_cs_1, _psnr_cs_2, _psnr_cs_3

#######################################################################

def ssim_4d(image_3d, image_rec_3d):

    _ssim_key_1 = compute_ssim(image_3d[:, :, 0]*255, image_rec_3d[:, :, 0]*255)
    _ssim_cs_1 = compute_ssim(image_3d[:, :, 1]*255, image_rec_3d[:, :, 1]*255)
    _ssim_cs_2 = compute_ssim(image_3d[:, :, 2]*255, image_rec_3d[:, :, 2]*255)
    _ssim_cs_3 = compute_ssim(image_3d[:, :, 3]*255, image_rec_3d[:, :, 3]*255)

    return _ssim_key_1, _ssim_cs_1, _ssim_cs_2, _ssim_cs_3

#######################################################################

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

#######################################################################

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

#######################################################################

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

#######################################################################




class FCNet_key(keras.Model):

    def __init__(self, key_size, block_size):##key_size是关键帧采样率，block_size是采样区域的大小
        super(FCNet_key, self).__init__()

        # z1: [b, h, w, 1] => [b, h/block_size, w/blcok_size, key_size] 采样了，filters相当于通道数
        self.conv1 = layers.Conv2D(filters = key_size, kernel_size = block_size, strides = (block_size, block_size), padding = 'valid')
        # z2: [b, h/block_size, w/blcok_size, key_size] => [b, h/block_size, w/blcok_size, block_size ** 2]
        self.conv2 = layers.Conv2D(filters = block_size ** 2, kernel_size = 1, strides = (1, 1), padding = 'valid')##大小为1的卷积核就是用来升维的
        # z2: [b, h/block_size, w/blcok_size, block_size ** 2] =>[b, h, w, 1]
        # self.reshape = tf.nn.depth_to_space(block_size)
        # z3: [b, h, w, 1] => [b, h, w, 1]
        self.conv3 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        # z4-z13
        self.conv4 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv5 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv6 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv7 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv8 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv9 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv10 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv11 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv12 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        self.conv13 = layers.Conv2D(filters = 64, kernel_size = 3, strides = (1, 1), padding = 'same')
        # z14
        self.conv14 = layers.Conv2D(filters = 1, kernel_size = 3, strides = (1, 1), padding = 'same')


    def call(self, inputs,  training=None):

        z1 = self.conv1(inputs)
        z2 = self.conv2(z1)
        z2 = tf.nn.depth_to_space(z2, block_size)

        z3 = self.conv3(z2)
        z3 = tf.nn.relu(z3)

        z4 = self.conv4(z3)
        z4 = tf.nn.relu(z4)

        z5 = self.conv5(z4)
        z5 = tf.nn.relu(z5)

        z6 = self.conv6(z5)
        z6 = tf.nn.relu(z6)

        z7 = self.conv7(z6)
        z7 = tf.nn.relu(z7)

        z8 = self.conv8(z7)
        z8 = tf.nn.relu(z8)

        z9 = self.conv9(z8)
        z9 = tf.nn.relu(z9)

        z10 = self.conv10(z9)
        z10 = tf.nn.relu(z10)

        z11 = self.conv11(z10)
        z11 = tf.nn.relu(z11)

        z12 = self.conv12(z11)
        z12 = tf.nn.relu(z12)

        z13 = self.conv13(z12)
        z13 = tf.nn.relu(z13)

        z14 = self.conv14(z13)


        return z14

    #######################################################################

class FCNet_cs_1(keras.Model):

    def __init__(self, cs_size, block_size):
        super(FCNet_cs_1, self).__init__()

        # z1: [b, h, w, 1] => [b, h/block_size, w/blcok_size, key_size]
        self.conv1 = layers.Conv2D(filters=cs_size, kernel_size=block_size, strides=(block_size, block_size),
                                   padding='valid')
        # z2: [b, h/block_size, w/blcok_size, key_size] => [b, h/block_size, w/blcok_size, block_size ** 2]
        self.conv2 = layers.Conv2D(filters=block_size ** 2, kernel_size=1, strides=(1, 1), padding='valid')
        # z2: [b, h/block_size, w/blcok_size, block_size ** 2] =>[b, h, w, 1]
        # self.reshape = tf.nn.depth_to_space(block_size)
        # z3: [b, h, w, 1] => [b, h, w, 1]
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        # z4-z13
        self.conv4 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv5 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv6 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv7 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv8 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv9 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv10 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv11 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv12 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv13 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        # z14
        self.conv14 = layers.Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same')

    def call(self, inputs, training=None):
        z1 = self.conv1(inputs)
        z2 = self.conv2(z1)
        z2 = tf.nn.depth_to_space(z2, block_size)

        z3 = self.conv3(z2)
        z3 = tf.nn.relu(z3)

        z4 = self.conv4(z3)
        z4 = tf.nn.relu(z4)

        z5 = self.conv5(z4)
        z5 = tf.nn.relu(z5)

        z6 = self.conv6(z5)
        z6 = tf.nn.relu(z6)

        z7 = self.conv7(z6)
        z7 = tf.nn.relu(z7)

        z8 = self.conv8(z7)
        z8 = tf.nn.relu(z8)

        z9 = self.conv9(z8)
        z9 = tf.nn.relu(z9)

        z10 = self.conv10(z9)
        z10 = tf.nn.relu(z10)

        z11 = self.conv11(z10)
        z11 = tf.nn.relu(z11)

        z12 = self.conv12(z11)
        z12 = tf.nn.relu(z12)

        z13 = self.conv13(z12)
        z13 = tf.nn.relu(z13)

        z14 = self.conv14(z13)

        return z14

 #######################################################################

class FCNet_cs_2(keras.Model):

    def __init__(self, cs_size, block_size):
        super(FCNet_cs_2, self).__init__()

        # z1: [b, h, w, 1] => [b, h/block_size, w/blcok_size, key_size]
        self.conv1 = layers.Conv2D(filters=cs_size, kernel_size=block_size, strides=(block_size, block_size),
                                   padding='valid')
        # z2: [b, h/block_size, w/blcok_size, key_size] => [b, h/block_size, w/blcok_size, block_size ** 2]
        self.conv2 = layers.Conv2D(filters=block_size ** 2, kernel_size=1, strides=(1, 1), padding='valid')
        # z2: [b, h/block_size, w/blcok_size, block_size ** 2] =>[b, h, w, 1]
        # self.reshape = tf.nn.depth_to_space(block_size)
        # z3: [b, h, w, 1] => [b, h, w, 1]
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        # z4-z13
        self.conv4 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv5 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv6 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv7 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv8 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv9 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv10 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv11 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv12 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv13 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        # z14
        self.conv14 = layers.Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same')

    def call(self, inputs,  training=None):
        z1 = self.conv1(inputs)
        z2 = self.conv2(z1)
        z2 = tf.nn.depth_to_space(z2, block_size)

        z3 = self.conv3(z2)
        z3 = tf.nn.relu(z3)

        z4 = self.conv4(z3)
        z4 = tf.nn.relu(z4)

        z5 = self.conv5(z4)
        z5 = tf.nn.relu(z5)

        z6 = self.conv6(z5)
        z6 = tf.nn.relu(z6)

        z7 = self.conv7(z6)
        z7 = tf.nn.relu(z7)

        z8 = self.conv8(z7)
        z8 = tf.nn.relu(z8)

        z9 = self.conv9(z8)
        z9 = tf.nn.relu(z9)

        z10 = self.conv10(z9)
        z10 = tf.nn.relu(z10)

        z11 = self.conv11(z10)
        z11 = tf.nn.relu(z11)

        z12 = self.conv12(z11)
        z12 = tf.nn.relu(z12)

        z13 = self.conv13(z12)
        z13 = tf.nn.relu(z13)

        z14 = self.conv14(z13)

        return z14

 #######################################################################

class FCNet_cs_3(keras.Model):

    def __init__(self, cs_size, block_size):
        super(FCNet_cs_3, self).__init__()

        # z1: [b, h, w, 1] => [b, h/block_size, w/blcok_size, key_size]
        self.conv1 = layers.Conv2D(filters=cs_size, kernel_size=block_size, strides=(block_size, block_size),
                                   padding='valid')
        # z2: [b, h/block_size, w/blcok_size, key_size] => [b, h/block_size, w/blcok_size, block_size ** 2]
        self.conv2 = layers.Conv2D(filters=block_size ** 2, kernel_size=1, strides=(1, 1), padding='valid')
        # z2: [b, h/block_size, w/blcok_size, block_size ** 2] =>[b, h, w, 1]
        # self.reshape = tf.nn.depth_to_space(block_size)
        # z3: [b, h, w, 1] => [b, h, w, 1]
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        # z4-z13
        self.conv4 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv5 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv6 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv7 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv8 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv9 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv10 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv11 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv12 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        self.conv13 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')
        # z14
        self.conv14 = layers.Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding='same')

    def call(self, inputs,  training=None):
        z1 = self.conv1(inputs)
        z2 = self.conv2(z1)
        z2 = tf.nn.depth_to_space(z2, block_size)

        z3 = self.conv3(z2)
        z3 = tf.nn.relu(z3)

        z4 = self.conv4(z3)
        z4 = tf.nn.relu(z4)

        z5 = self.conv5(z4)
        z5 = tf.nn.relu(z5)

        z6 = self.conv6(z5)
        z6 = tf.nn.relu(z6)

        z7 = self.conv7(z6)
        z7 = tf.nn.relu(z7)

        z8 = self.conv8(z7)
        z8 = tf.nn.relu(z8)

        z9 = self.conv9(z8)
        z9 = tf.nn.relu(z9)

        z10 = self.conv10(z9)
        z10 = tf.nn.relu(z10)

        z11 = self.conv11(z10)
        z11 = tf.nn.relu(z11)

        z12 = self.conv12(z11)
        z12 = tf.nn.relu(z12)

        z13 = self.conv13(z12)
        z13 = tf.nn.relu(z13)

        z14 = self.conv14(z13)

        return z14

 #######################################################################

class FCNet(keras.Model):

    def __init__(self, key_size, cs_size, block_size):
        super(FCNet, self).__init__()
        self.key = FCNet_key(key_size, block_size)
        self.cs1 = FCNet_cs_1(cs_size, block_size)
        self.cs2 = FCNet_cs_2(cs_size, block_size)
        self.cs3 = FCNet_cs_3(cs_size, block_size)

    def call(self, inputs, training=None):
        key_data = tf.expand_dims(inputs[:, :, :, 0], -1)
        cs_1_data = tf.expand_dims(inputs[:, :, :, 1], -1)
        cs_2_data = tf.expand_dims(inputs[:, :, :, 2], -1)
        cs_3_data = tf.expand_dims(inputs[:, :, :, 3], -1)

        rec_key = self.key(key_data)
        rec_cs_1 = self.cs1(cs_1_data)
        rec_cs_2 = self.cs2(cs_2_data)
        rec_cs_3 = self.cs3(cs_3_data)

        rec = tf.concat([rec_key, rec_cs_1, rec_cs_2, rec_cs_3], axis=-1)

        return rec