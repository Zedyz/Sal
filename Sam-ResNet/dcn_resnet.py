from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.keras import backend as K

# Force channels-first so the old Theano weights match
K.set_image_data_format('channels_first')

from tensorflow.keras.layers import (
    Input, Activation, Conv2D, MaxPooling2D, ZeroPadding2D,
    BatchNormalization, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file

# (Optional) If you need real concatenation layers, import:
# from tensorflow.keras.layers import Concatenate

TH_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/releases/download/'
    'v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """Identity block without a stride change."""
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1  # channels-first

    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    # 1x1 conv
    x = Conv2D(
        filters=nb_filter1, kernel_size=(1, 1),
        name=conv_name_base + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # main conv
    x = Conv2D(
        filters=nb_filter2, kernel_size=(kernel_size, kernel_size),
        padding='same', name=conv_name_base + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1 conv
    x = Conv2D(
        filters=nb_filter3, kernel_size=(1, 1),
        name=conv_name_base + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # Elementwise sum: x + input_tensor
    x = Add(name=f'{conv_name_base}_add')([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """Convolution block that changes dimension with strides."""
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1  # channels-first

    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    # 1x1 conv with strides
    x = Conv2D(
        filters=nb_filter1, kernel_size=(1, 1), strides=strides,
        name=conv_name_base + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # main conv
    x = Conv2D(
        filters=nb_filter2, kernel_size=(kernel_size, kernel_size),
        padding='same', name=conv_name_base + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1 conv
    x = Conv2D(
        filters=nb_filter3, kernel_size=(1, 1),
        name=conv_name_base + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # Shortcut path
    shortcut = Conv2D(
        filters=nb_filter3, kernel_size=(1, 1), strides=strides,
        name=conv_name_base + '1'
    )(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # sum x + shortcut
    x = Add(name=f'{conv_name_base}_add')([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_atrous(input_tensor, kernel_size, filters, stage, block, atrous_rate=(2, 2)):
    """Convolution block with dilation (atrous) instead of strides."""
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    # 1x1 conv
    x = Conv2D(
        filters=nb_filter1, kernel_size=(1, 1),
        name=conv_name_base + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # main conv with dilation
    x = Conv2D(
        filters=nb_filter2, kernel_size=(kernel_size, kernel_size),
        dilation_rate=atrous_rate, padding='same',
        name=conv_name_base + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1 conv
    x = Conv2D(
        filters=nb_filter3, kernel_size=(1, 1),
        name=conv_name_base + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # shortcut
    shortcut = Conv2D(
        filters=nb_filter3, kernel_size=(1, 1),
        name=conv_name_base + '1'
    )(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # sum
    x = Add(name=f'{conv_name_base}_add')([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block_atrous(input_tensor, kernel_size, filters, stage, block, atrous_rate=(2, 2)):
    """Identity block with dilation (no dimension change)."""
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    # 1x1
    x = Conv2D(
        filters=nb_filter1, kernel_size=(1, 1),
        name=conv_name_base + '2a'
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # main conv with dilation
    x = Conv2D(
        filters=nb_filter2, kernel_size=(kernel_size, kernel_size),
        dilation_rate=atrous_rate,
        padding='same', name=conv_name_base + '2b'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 1x1
    x = Conv2D(
        filters=nb_filter3, kernel_size=(1, 1),
        name=conv_name_base + '2c'
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # sum
    x = Add(name=f'{conv_name_base}_add')([x, input_tensor])
    x = Activation('relu')(x)
    return x


def dcn_resnet(input_tensor=None):
    """
    Build a DCN-ResNet-like model with dilation in the last layers.
    Using channels-first => input_shape=(3, None, None).
    """
    # Make sure your data actually has shape (batch,3,H,W).
    input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 1  # channels-first

    # conv_1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # conv_2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # conv_3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv_4 (dilated)
    x = conv_block_atrous(x, 3, [256, 256, 1024], stage=4, block='a', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='b', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='c', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='d', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='e', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='f', atrous_rate=(2, 2))

    # conv_5 (more dilation)
    x = conv_block_atrous(x, 3, [512, 512, 2048], stage=5, block='a', atrous_rate=(4, 4))
    x = identity_block_atrous(x, 3, [512, 512, 2048], stage=5, block='b', atrous_rate=(4, 4))
    x = identity_block_atrous(x, 3, [512, 512, 2048], stage=5, block='c', atrous_rate=(4, 4))

    # Build the model
    model = Model(inputs=img_input, outputs=x)

    #--------------------------------------------------------------------------------------
    # Attempt partial weight loading
    # If your code REALLY needs these old Theano weights, load them in "by_name" mode
    # so shape mismatches or renamed layers get skipped. This avoids the shape error.
    #
    # If you do NOT need them, comment out or remove the load_weights line.
    #--------------------------------------------------------------------------------------
    weights_path = get_file(
        'resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
        TH_WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        md5_hash='f64f049c92468c9affcd44b0976cdafe'
    )

    try:
        # "by_name=True" tries to match layers by name, "skip_mismatch=True" ignores shape mismatches
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("Old Theano-based weights partially loaded (by_name, skipping mismatches).")
    except:
        print("Could not load old Theano weights. Using random initialization.")

    return model
