import tensorflow as tf
import numpy as np
import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d



def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

##############################################################################
##############################################################################

def count_dataset(filename):
    c = 0
    for _ in tf.python_io.tf_record_iterator(filename):
        c += 1
    print('conter')
    return c

##############################################################################
##############################################################################

def parse_function(example_proto):
    features = {
        'data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.float32)
    label = tf.decode_raw(parsed_features['label'], tf.float32)
    # data = tf.reshape(data,(160,160,3))
    # label = tf.reshape(label,(160,160,3))
    data = tf.reshape(data, (160, 160, 4))
    label = tf.reshape(label, (160, 160, 4))

    # label = tf.reshape(label, (160, 160, 1))

    return data, label

##############################################################################
##############################################################################

def _psnr(image,image_rec):
    diff = image - image_rec
    diff = diff.flatten('C')
    # diff = diff.eval()
    # a, b = diff.shape
    # diff = diff.reshape((a*b,1))
    rmse = math.sqrt(np.mean(diff ** 2.))
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr

##############################################################################
##############################################################################

def _psnr_3d(image_3d,image_rec_3d):

    _psnr_key_1 = _psnr(image_3d[:,:,0],image_rec_3d[:,:,0])
    _psnr_key_2 = _psnr(image_3d[:,:,2],image_rec_3d[:,:,2])
    _psnr_cs = _psnr(image_3d[:,:,1],image_rec_3d[:,:,1])

    return _psnr_key_1,_psnr_cs,_psnr_key_2

##############################################################################
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


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


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
##############################################################################

def _psnr_2d(image_3d,image_rec_3d):

    _psnr_key_1 = _psnr(image_3d[:,:,0],image_rec_3d[:,:,0])
    _psnr_cs = _psnr(image_3d[:,:,1],image_rec_3d[:,:,1])

    return _psnr_key_1,_psnr_cs

##############################################################################
def _ssim_4d(image_3d, image_rec_3d):

    _ssim_key_1 = compute_ssim(image_3d[:, :, 0]*255, image_rec_3d[:, :, 0]*255)
    _ssim_cs_1 = compute_ssim(image_3d[:, :, 1]*255, image_rec_3d[:, :, 1]*255)
    _ssim_cs_2 = compute_ssim(image_3d[:, :, 2]*255, image_rec_3d[:, :, 2]*255)
    _ssim_cs_3 = compute_ssim(image_3d[:, :, 3]*255, image_rec_3d[:, :, 3]*255)

    return _ssim_key_1, _ssim_cs_1, _ssim_cs_2, _ssim_cs_3

##############################################################################


def _psnr_4d(image_3d,image_rec_3d):

    _psnr_key_1 = _psnr(image_3d[:, :, 0],image_rec_3d[:, :, 0])
    _psnr_cs_1 = _psnr(image_3d[:, :, 1],image_rec_3d[:, :, 1])
    _psnr_cs_2 = _psnr(image_3d[:, :, 2], image_rec_3d[:, :, 2])
    _psnr_cs_3 = _psnr(image_3d[:, :, 3], image_rec_3d[:, :, 3])

    return _psnr_key_1,_psnr_cs_1, _psnr_cs_2, _psnr_cs_3

##############################################################################
##############################################################################

def collect_label(label):
    a = label
    return a

##############################################################################
##############################################################################

def _variable_with_weight_decay(name, shape, initializer, weight_decay_rate):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=True)
    if weight_decay_rate is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay_rate, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

##############################################################################
##############################################################################

def convolution(stage, kernel_shape, w_std, weight_decay_rate, input_data, stride, padding_method, relu, bn, is_training):
    """

    :param stage:
    :param kernel_shape:[h_stride, w_stride, input_channel, output_channel]
    :param w_std:
    :param weight_decay_rate:
    :param input_data: N*H*W*C
    :param stride:
    :param padding_method:
    :param biases_shape:
    :param relu:
    :return:
    """

    biases_shape = [kernel_shape[3]]
    conv_name = 'conv' + str(stage)
    with tf.variable_scope(conv_name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             kernel_shape,
                                             tf.truncated_normal_initializer(stddev = w_std),
                                             weight_decay_rate)
        conv = tf.nn.conv2d(input_data, kernel, stride, padding_method)
        biases = _variable_with_weight_decay('biases',
                                             biases_shape,
                                             tf.constant_initializer(0.0),
                                             weight_decay_rate)
        activation = tf.nn.bias_add(conv, biases)

        if relu == 1:
            if bn == 0:
                activation = tf.nn.relu(activation)
            else:
                activation = tf.layers.batch_normalization(activation, training = is_training)
                activation = tf.nn.relu(activation)
    stage = stage + 1
    return activation, stage

##############################################################################
##############################################################################

def convolution_after(stage, kernel_shape, w_std, weight_decay_rate, input_data, stride, padding_method, relu, bn, is_training):
    """

    :param stage:
    :param kernel_shape:[h_stride, w_stride, input_channel, output_channel]
    :param w_std:
    :param weight_decay_rate:
    :param input_data: N*H*W*C
    :param stride:
    :param padding_method:
    :param biases_shape:
    :param relu:
    :return:
    """

    biases_shape = [kernel_shape[3]]
    conv_name = 'after_coooooonv' + str(stage)
    with tf.variable_scope(conv_name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             kernel_shape,
                                             tf.truncated_normal_initializer(stddev = w_std),
                                             weight_decay_rate)
        conv = tf.nn.conv2d(input_data, kernel, stride, padding_method)
        biases = _variable_with_weight_decay('biases',
                                             biases_shape,
                                             tf.constant_initializer(0.0),
                                             weight_decay_rate)
        activation = tf.nn.bias_add(conv, biases)

        if relu == 1:
            if bn == 0:
                activation = tf.nn.relu(activation)
            else:
                activation = tf.layers.batch_normalization(activation, training = is_training)
                activation = tf.nn.relu(activation)
    stage = stage + 1
    return activation, stage

##############################################################################
##############################################################################

def Inception_module(data, stage, weight_decay_rate, is_training):

    conv_1, stage = convolution_after(stage, [1, 1, data.shape[3], 16], 0.1, weight_decay_rate, data, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)
    conv_2, stage = convolution_after(stage, [3, 3, data.shape[3], 16], 0.1, weight_decay_rate, data, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)
    conv_3, stage = convolution_after(stage, [5, 5, data.shape[3], 16], 0.1, weight_decay_rate, data, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)
    prediction = tf.concat([conv_1, conv_2, conv_3], 3)

    return prediction, prediction.shape[3], stage

##############################################################################
##############################################################################

def myloss(prediction, label,reg_loss ):
    ##############################################################################3
    without_regular_loss = tf.losses.mean_squared_error(prediction, label)
    loss = without_regular_loss
    # loss_1 = without_regular_loss
    # loss_2 = tf.add_n(tf.get_collection('losses'))
    # loss = loss_1 + loss_2
    ##############################################################################
    # k = 10
    # prediction_key_1 = prediction[:,:,:,0:1]
    # prediction_key_2 = prediction[:,:,:,2:3]
    # prediction_cs = prediction[:,:,:,1:2]
    # label_key_1 = label[:,:,:,0:1]
    # label_key_2 = label[:,:,:,2:3]
    # label_cs = label[:,:,:,1:2]
    # prediction_key = tf.concat([prediction_key_1,prediction_key_2],3)
    # label_key = tf.concat([label_key_1, label_key_2], 3)
    # loss_key = tf.losses.mean_squared_error(prediction_key, label_key)
    # print(loss_key)
    # loss_cs = tf.losses.mean_squared_error(prediction_cs, label_cs)
    # loss = (k*loss_key + loss_cs)


    return loss

##############################################################################
##############################################################################

def FCNet_phi_trainable_key_1(data, block_size, subrate, weight_decay_rate, is_training):
    y_size = math.ceil(subrate * (block_size ** 2))

    stage = 999
    conv_phi, xxx = convolution(stage, [block_size, block_size, 1, y_size], 0.01, weight_decay_rate, data, [1, block_size, block_size, 1], 'VALID', relu = 0, bn = 0, is_training = is_training)

    stage = 0
    conv0, stage = convolution(stage, [1, 1, y_size, block_size * block_size], 0.01, weight_decay_rate, conv_phi,
                               [1, 1, 1, 1],
                               'VALID', relu = 0, bn = 1, is_training = is_training)

    conv0_reshape = tf.nn.depth_to_space(conv0, block_size, name='conv0_reshape')

    conv, stage = convolution(stage, [3, 3, 1, 64], 0.1, weight_decay_rate, conv0_reshape, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution(stage, [3, 3, 64, 1], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    return conv

##############################################################################
##############################################################################

def FCNet_phi_trainable_nonkey_1(data, block_size, subrate, weight_decay_rate, is_training):
    y_size = math.ceil(subrate * (block_size ** 2))

    stage = 19999
    conv_phi, xxx = convolution_after(stage, [block_size, block_size, 1, y_size], 0.01, weight_decay_rate, data, [1, block_size, block_size, 1], 'VALID', relu = 0, bn = 0, is_training = is_training)

    stage = 20000
    conv0, stage = convolution_after(stage, [1, 1, y_size, block_size * block_size], 0.01, weight_decay_rate, conv_phi,
                               [1, 1, 1, 1],
                               'VALID', relu = 1, bn = 0, is_training = is_training)

    conv0_reshape = tf.nn.depth_to_space(conv0, block_size, name='conv_after_0_reshape')

    conv, stage = convolution_after(stage, [3, 3, 1, 64], 0.1, weight_decay_rate, conv0_reshape, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 1], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    return conv

##############################################################################
##############################################################################

def FCNet_phi_trainable_nonkey_2(data, block_size, subrate, weight_decay_rate, is_training):
    y_size = math.ceil(subrate * (block_size ** 2))

    stage = 29999
    conv_phi, xxx = convolution_after(stage, [block_size, block_size, 1, y_size], 0.01, weight_decay_rate, data, [1, block_size, block_size, 1], 'VALID', relu = 0, bn = 0, is_training = is_training)

    stage = 30000
    conv0, stage = convolution_after(stage, [1, 1, y_size, block_size * block_size], 0.01, weight_decay_rate, conv_phi,
                               [1, 1, 1, 1],
                               'VALID', relu = 1, bn = 0, is_training = is_training)

    conv0_reshape = tf.nn.depth_to_space(conv0, block_size, name='conv_after_0_reshape')

    conv, stage = convolution_after(stage, [3, 3, 1, 64], 0.1, weight_decay_rate, conv0_reshape, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 1], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    return conv

##############################################################################
##############################################################################

def FCNet_phi_trainable_nonkey_3(data, block_size, subrate, weight_decay_rate, is_training):
    y_size = math.ceil(subrate * (block_size ** 2))

    stage = 39999
    conv_phi, xxx = convolution_after(stage, [block_size, block_size, 1, y_size], 0.01, weight_decay_rate, data, [1, block_size, block_size, 1], 'VALID', relu = 0, bn = 0, is_training = is_training)

    stage = 40000
    conv0, stage = convolution_after(stage, [1, 1, y_size, block_size * block_size], 0.01, weight_decay_rate, conv_phi,
                               [1, 1, 1, 1],
                               'VALID', relu = 1, bn = 0, is_training = is_training)

    conv0_reshape = tf.nn.depth_to_space(conv0, block_size, name='conv_after_0_reshape')

    conv, stage = convolution_after(stage, [3, 3, 1, 64], 0.1, weight_decay_rate, conv0_reshape, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 64], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    conv, stage = convolution_after(stage, [3, 3, 64, 1], 0.1, weight_decay_rate, conv, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0, is_training = is_training)

    return conv

##############################################################################
##############################################################################

def DINet_2 (data, block_size, subrate_key, subrate_cs, weight_decay_rate, is_training):
    '''

    :param data:
    :param data_shape:
    :param block_size:
    :param subrate:
    :param weight_decay:
    :return:
    '''

    stage = 9999999
    key = data[:,:,:,0:1]
    key_rec = FCNet_phi_trainable_key_1(key, block_size, subrate_key, weight_decay_rate, is_training)
    cs = data[:, :, :, 1:2]
    cs_conv = FCNet_phi_trainable_nonkey_1(cs, block_size, subrate_cs, weight_decay_rate,is_training)


    # key_conv_1, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)
    inception_1_concat = tf.concat([key_rec,cs_conv],3)
    inception_1, num_concat_1, stage = Inception_module(inception_1_concat, stage, weight_decay_rate, is_training)
    inception_1_conv, stage = convolution_after(stage, [3, 3, num_concat_1, 1], 0.1, weight_decay_rate, inception_1, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_2, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)
    inception_2_concat = tf.concat([key_rec, inception_1_conv], 3)
    inception_2, num_concat_2, stage = Inception_module(inception_2_concat, stage, weight_decay_rate, is_training)
    inception_2_conv, stage = convolution_after(stage, [3, 3, num_concat_2, 1], 0.1, weight_decay_rate,inception_2,[1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_3, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0,is_training = is_training)
    inception_3_concat = tf.concat([key_rec, inception_2_conv], 3)
    inception_3, num_concat_3, stage = Inception_module(inception_3_concat, stage, weight_decay_rate, is_training)
    inception_3_conv, stage = convolution_after(stage, [3, 3, num_concat_3, 1], 0.1, weight_decay_rate,inception_3,[1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_4, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0,is_training = is_training)
    inception_4_concat = tf.concat([key_rec, inception_3_conv], 3)
    inception_4, num_concat_4, stage = Inception_module(inception_4_concat, stage, weight_decay_rate, is_training)
    inception_4_conv, stage = convolution_after(stage, [3, 3, num_concat_4, 1], 0.1, weight_decay_rate,inception_4,[1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_5, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0,is_training = is_training)
    inception_5_concat = tf.concat([key_rec, inception_4_conv], 3)
    inception_5, num_concat_5, stage = Inception_module(inception_5_concat, stage, weight_decay_rate, is_training)
    inception_5_conv, stage = convolution_after(stage, [3, 3, num_concat_5, 1], 0.1, weight_decay_rate, inception_5, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_6, stage = convolution_after(stage, [3, 3, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0,is_training = is_training)
    output = inception_5_conv + cs_conv
    output = tf.concat([key_rec,output],3)

    return output

##############################################################################
##############################################################################

def DINet_4 (data, block_size, subrate_key, subrate_cs, weight_decay_rate, is_training):
    '''

    :param data:
    :param data_shape:
    :param block_size:
    :param subrate:
    :param weight_decay:
    :return:
    '''
    num_output = 4
    stage = 9999999
    key = data[:,:,:,0:1]
    key_rec = FCNet_phi_trainable_key_1(key, block_size, subrate_key, weight_decay_rate, is_training)
    nonkey_1 = data[:, :, :, 1:2]
    nonkey_1_conv = FCNet_phi_trainable_nonkey_1(nonkey_1, block_size, subrate_cs, weight_decay_rate,is_training)
    nonkey_2 = data[:, :, :, 2:3]
    nonkey_2_conv = FCNet_phi_trainable_nonkey_2(nonkey_2, block_size, subrate_cs, weight_decay_rate, is_training)
    nonkey_3 = data[:, :, :, 3:4]
    nonkey_3_conv = FCNet_phi_trainable_nonkey_3(nonkey_3, block_size, subrate_cs, weight_decay_rate, is_training)
    nonkey_rec = tf.concat([nonkey_1_conv, nonkey_2_conv, nonkey_3_conv],3)


    # key_conv_1, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)
    inception_1_concat = tf.concat([key_rec, nonkey_rec],3)
    inception_1, num_concat_1, stage = Inception_module(inception_1_concat, stage, weight_decay_rate, is_training)
    inception_1_conv, stage = convolution_after(stage, [3, 3, num_concat_1, num_output - 1], 0.1, weight_decay_rate, inception_1, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_2, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)
    inception_2_concat = tf.concat([key_rec, inception_1_conv], 3)
    inception_2, num_concat_2, stage = Inception_module(inception_2_concat, stage, weight_decay_rate, is_training)
    inception_2_conv, stage = convolution_after(stage, [3, 3, num_concat_2, num_output - 1], 0.1, weight_decay_rate,inception_2,[1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_3, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0,is_training = is_training)
    inception_3_concat = tf.concat([key_rec, inception_2_conv], 3)
    inception_3, num_concat_3, stage = Inception_module(inception_3_concat, stage, weight_decay_rate, is_training)
    inception_3_conv, stage = convolution_after(stage, [3, 3, num_concat_3, num_output - 1], 0.1, weight_decay_rate,inception_3,[1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_4, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0,is_training = is_training)
    inception_4_concat = tf.concat([key_rec, inception_3_conv], 3)
    inception_4, num_concat_4, stage = Inception_module(inception_4_concat, stage, weight_decay_rate, is_training)
    inception_4_conv, stage = convolution_after(stage, [3, 3, num_concat_4, num_output - 1], 0.1, weight_decay_rate,inception_4,[1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_5, stage = convolution_after(stage, [1, 1, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0,is_training = is_training)
    inception_5_concat = tf.concat([key_rec, inception_4_conv], 3)
    inception_5, num_concat_5, stage = Inception_module(inception_5_concat, stage, weight_decay_rate, is_training)
    inception_5_conv, stage = convolution_after(stage, [3, 3, num_concat_5, num_output - 1], 0.1, weight_decay_rate, inception_5, [1, 1, 1, 1], 'SAME', relu = 0, bn = 0, is_training = is_training)

    # key_conv_6, stage = convolution_after(stage, [3, 3, 1, 1], 0.1, weight_decay_rate, key_rec, [1, 1, 1, 1], 'SAME', relu = 1, bn = 0,is_training = is_training)

    output = inception_5_conv + nonkey_rec
    output = tf.concat([key_rec,output],3)


    return output

