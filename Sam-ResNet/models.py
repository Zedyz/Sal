from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Lambda, Concatenate, Conv2D
)

from config import nb_timestep, shape_r_gt, shape_c_gt, upsampling_factor, nb_gaussian
# from config import b_s, nb_timestep, shape_r_gt, shape_c_gt, shape_r_out, shape_c_out, upsampling_factor, nb_gaussian
# e.g.:
# b_s = 8
# nb_timestep = 5
# shape_r_gt = 14
# shape_c_gt = 14
# shape_r_out = 14
# shape_c_out = 14
# upsampling_factor = 2
# nb_gaussian = 5

from dcn_vgg import dcn_vgg
from dcn_resnet import dcn_resnet
from gaussian_prior import LearningPrior
from attentive_convlstm import AttentiveConvLSTM


# -------------------------------------------------------------------------
# 1) Helper Lambdas
# -------------------------------------------------------------------------
def repeat(x):
    """
    Repeats 'x' along the second dimension (nb_timestep).
    For example, x shape = (b_s, 512, shape_r_gt, shape_c_gt).
    We reshape and tile so final shape = (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt).
    """
    # Flatten each batch, repeat nb_timestep times, then reshape back
    # x is 4D => (b_s, 512, shape_r_gt, shape_c_gt)
    # We want => (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt)
    x_flat = tf.reshape(x, (K.shape(x)[0], -1))  # shape => (b_s, 512*shape_r_gt*shape_c_gt)
    x_rep = tf.tile(x_flat, [1, nb_timestep])  # (b_s, nb_timestep * 512*shape_r_gt*shape_c_gt)
    return tf.reshape(
        x_rep,
        (K.shape(x)[0], nb_timestep, 512, shape_r_gt, shape_c_gt)
    )


def repeat_shape(s):
    # s is (b_s, 512, shape_r_gt, shape_c_gt)
    # we want => (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt)
    return (s[0], nb_timestep, s[1], s[2], s[3])


def upsampling(x):
    """
    Bilinear upsampling via tf.image.resize or UpSampling2D.
    x shape = (b_s, 1, shape_r_out, shape_c_out) if channels-first
    or (b_s, shape_r_out, shape_c_out, 1) if channels-last.
    We'll assume channels-first for consistency with Theano style.
    """
    # We'll assume (b_s, 1, shape_r_out, shape_c_out)
    # We want to upsample by 'upsampling_factor' using bilinear
    # => final shape: (b_s, 1, shape_r_out*factor, shape_c_out*factor)
    # We can do a quick transpose to channels-last, resize, transpose back
    # Alternatively, use UpSampling2D if you just want nearest neighbor.
    # Here is a custom approach:
    factor = upsampling_factor

    # (b_s, 1, H, W) => (b_s, H, W, 1)
    x_tchw = tf.transpose(x, [0, 2, 3, 1])
    # Bilinear resize
    new_height = K.shape(x_tchw)[1] * factor
    new_width = K.shape(x_tchw)[2] * factor
    x_resized = tf.image.resize(x_tchw, [new_height, new_width], method='bilinear')
    # transpose back => (b_s, 1, H*factor, W*factor)
    x_bcHW = tf.transpose(x_resized, [0, 3, 1, 2])
    return x_bcHW


def upsampling_shape(s):
    # s => (b_s, 1, shape_r_out, shape_c_out)
    return (s[0], s[1], s[2] * upsampling_factor, s[3] * upsampling_factor)


# -------------------------------------------------------------------------
# 2) Loss Functions
# -------------------------------------------------------------------------
def kl_divergence(y_true, y_pred):
    """
    KL-Div vs. normalized maps.
    Expects channels-first shape: (b_s, 1, shape_r_out, shape_c_out)
    or possibly (b_s, something, shape_r_out, shape_c_out).
    """
    # 1) Normalize y_pred by its max
    max_y_pred = tf.reduce_max(y_pred, axis=[2, 3], keepdims=True)
    y_pred = y_pred / (max_y_pred + K.epsilon())

    # 2) Sum over spatial dims for y_true & y_pred
    sum_y_true = tf.reduce_sum(y_true, axis=[2, 3], keepdims=True)
    sum_y_pred = tf.reduce_sum(y_pred, axis=[2, 3], keepdims=True)
    y_true_norm = y_true / (sum_y_true + K.epsilon())
    y_pred_norm = y_pred / (sum_y_pred + K.epsilon())

    # KL = y_true * log(y_true / y_pred)
    # sum over last two dims
    kl_map = y_true_norm * K.log(y_true_norm / (y_pred_norm + K.epsilon()) + K.epsilon())
    return 10.0 * tf.reduce_sum(kl_map, axis=[2, 3])


def correlation_coefficient(y_true, y_pred):
    """
    -2 * correlation coefficient.
    Both maps are normalized to sum=1 before correlation is computed.
    """
    # normalize y_pred by max
    max_y_pred = tf.reduce_max(y_pred, axis=[2, 3], keepdims=True)
    y_pred = y_pred / (max_y_pred + K.epsilon())

    # sum
    sum_y_true = tf.reduce_sum(y_true, axis=[2, 3], keepdims=True)
    sum_y_pred = tf.reduce_sum(y_pred, axis=[2, 3], keepdims=True)
    y_true_norm = y_true / (sum_y_true + K.epsilon())
    y_pred_norm = y_pred / (sum_y_pred + K.epsilon())

    # flatten
    y_true_f = tf.reshape(y_true_norm, (K.shape(y_true_norm)[0], K.shape(y_true_norm)[1], -1))
    y_pred_f = tf.reshape(y_pred_norm, (K.shape(y_pred_norm)[0], K.shape(y_pred_norm)[1], -1))

    # Now compute correlation per batch
    # sum_xy, sum_x, sum_y, sum_x2, sum_y2
    sum_xy = tf.reduce_sum(y_true_f * y_pred_f, axis=2)
    sum_x = tf.reduce_sum(y_true_f, axis=2)
    sum_y = tf.reduce_sum(y_pred_f, axis=2)
    sum_x2 = tf.reduce_sum(tf.square(y_true_f), axis=2)
    sum_y2 = tf.reduce_sum(tf.square(y_pred_f), axis=2)

    N = tf.cast(K.shape(y_true_f)[2], K.floatx())  # number of pixels
    num = sum_xy - ((sum_x * sum_y) / N)
    den = tf.sqrt((sum_x2 - tf.square(sum_x) / N) * (sum_y2 - tf.square(sum_y) / N) + K.epsilon())

    corr = num / (den + K.epsilon())
    return -2.0 * corr  # we often want to minimize => negative


def nss(y_true, y_pred):
    """
    Normalized Scanpath Saliency
    y_true is binary fixations, y_pred is predicted saliency
    """
    # normalize y_pred by max
    max_y_pred = tf.reduce_max(y_pred, axis=[2, 3], keepdims=True)
    y_pred_norm = y_pred / (max_y_pred + K.epsilon())

    # flatten
    b_s = K.shape(y_pred_norm)[0]
    c_s = K.shape(y_pred_norm)[1]
    y_pred_flat = tf.reshape(y_pred_norm, (b_s, c_s, -1))

    # mean, std
    mean_pred = tf.reduce_mean(y_pred_flat, axis=-1, keepdims=True)
    std_pred = tf.math.reduce_std(y_pred_flat, axis=-1, keepdims=True) + K.epsilon()
    # shape => (b_s, c_s, 1)
    # broadcast back
    y_pred_centered = (y_pred_flat - mean_pred) / std_pred
    # shape => (b_s, c_s, shape_r_out*shape_c_out)
    # reshape to 4D again
    y_pred_centered_4d = tf.reshape(y_pred_centered, K.shape(y_pred_norm))

    # NSS = sum( fix * normalized_map ) / sum(fix)
    fix_sum = tf.reduce_sum(y_true, axis=[2, 3]) + K.epsilon()  # shape => (b_s, c_s)
    numerator = tf.reduce_sum(y_true * y_pred_centered_4d, axis=[2, 3])
    return -(numerator / fix_sum)


# -------------------------------------------------------------------------
# 3) sam_vgg & sam_resnet
# -------------------------------------------------------------------------
def sam_vgg(x):
    """
    x[0] => image input
    x[1] => prior input
    """
    # 1) Build the VGG-based DCN
    dcn = dcn_vgg(input_tensor=x[0])  # returns a model
    dcn_feat = dcn.output  # shape => (b_s, 512, ?, ?), presumably

    # 2) Attentive ConvLSTM
    att_convlstm = Lambda(repeat, output_shape=repeat_shape)(dcn_feat)
    # att_convlstm shape => (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt)
    att_convlstm = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,
                                     nb_rows=3, nb_cols=3)(att_convlstm)
    # shape => (b_s, 512, shape_r_gt, shape_c_gt)

    # 3) Learned Prior (1)
    priors1 = LearningPrior(nb_gaussian=nb_gaussian)(x[1])  # shape => (b_s, nb_gaussian, height, width)
    concatenated = Concatenate(axis=1)([att_convlstm, priors1])
    # e.g. => (b_s, 512 + nb_gaussian, shape_r_gt, shape_c_gt)

    learned_priors1 = Conv2D(filters=512, kernel_size=(5, 5), padding='same',
                             activation='relu', dilation_rate=(4, 4))(concatenated)

    # 4) Learned Prior (2)
    priors2 = LearningPrior(nb_gaussian=nb_gaussian)(x[1])
    concatenated2 = Concatenate(axis=1)([learned_priors1, priors2])
    learned_priors2 = Conv2D(filters=512, kernel_size=(5, 5), padding='same',
                             activation='relu', dilation_rate=(4, 4))(concatenated2)

    # 5) Final conv => 1 channel
    outs = Conv2D(filters=1, kernel_size=(1, 1), activation='relu', padding='same')(learned_priors2)
    # shape => (b_s, 1, shape_r_gt, shape_c_gt)

    # 6) Upsampling
    outs_up = Lambda(upsampling, output_shape=upsampling_shape)(outs)

    # Return triple identical outputs (like original code)
    return [outs_up, outs_up, outs_up]


def sam_resnet(x):
    """
    x[0] => image input
    x[1] => prior input
    """
    # 1) ResNet-based DCN
    dcn = dcn_resnet(input_tensor=x[0])
    dcn_out = dcn.output  # shape => (b_s, 2048?, h, w)? But your code does a conv to reduce to 512

    conv_feat = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(dcn_out)
    # => (b_s, 512, h, w)

    # 2) Attentive ConvLSTM
    att_convlstm = Lambda(repeat, output_shape=repeat_shape)(conv_feat)
    att_convlstm = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,
                                     nb_rows=3, nb_cols=3)(att_convlstm)

    # 3) Learned prior (1)
    priors1 = LearningPrior(nb_gaussian=nb_gaussian)(x[1])  # (b_s, nb_gaussian, h, w)
    concatenated = Concatenate(axis=1)([att_convlstm, priors1])
    learned_priors1 = Conv2D(filters=512, kernel_size=(5, 5), dilation_rate=(4, 4),
                             padding='same', activation='relu')(concatenated)

    # 4) Learned prior (2)
    priors2 = LearningPrior(nb_gaussian=nb_gaussian)(x[1])
    concatenated2 = Concatenate(axis=1)([learned_priors1, priors2])
    learned_priors2 = Conv2D(filters=512, kernel_size=(5, 5), dilation_rate=(4, 4),
                             padding='same', activation='relu')(concatenated2)

    # 5) Final 1x1 conv => 1 channel
    outs = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(learned_priors2)
    # shape => (b_s, 1, h, w)

    # 6) Upsample
    outs_up = Lambda(upsampling, output_shape=upsampling_shape)(outs)

    return [outs_up, outs_up, outs_up]
