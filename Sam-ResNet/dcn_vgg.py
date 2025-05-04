from __future__ import print_function, absolute_import

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_file

TH_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
)

def dcn_vgg(input_tensor=None):
    """
    Builds a partial VGG-16-like network with dilation in the 5th block.
    Uses channels-first ordering: (3, height, width).
    """
    # Input shape: (3, None, None) if channels-first
    input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        # If user passes a tensor
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # conv_1
    x = Conv2D(filters=64, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(filters=64, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                     name='block1_pool')(x)

    # conv_2
    x = Conv2D(filters=128, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block2_conv1')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                     name='block2_pool')(x)

    # conv_3
    x = Conv2D(filters=256, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block3_conv1')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block3_conv2')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block3_conv3')(x)
    # Replacing what looks like an accidental Conv2D((2,2), strides=(2,2)) with a pooling layer:
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                     padding='same', name='block3_pool')(x)

    # conv_4
    x = Conv2D(filters=512, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block4_conv1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block4_conv2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block4_conv3')(x)
    # VGG standard block4 pool
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                     padding='same', name='block4_pool')(x)

    # conv_5 (dilated)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block5_conv1', dilation_rate=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block5_conv2', dilation_rate=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3),
               activation='relu', padding='same',
               name='block5_conv3', dilation_rate=(2, 2))(x)

    # Create model
    model = Model(inputs=img_input, outputs=x)

    # Load Theano-style weights (channels-first)
    weights_path = get_file(
        'vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
        TH_WEIGHTS_PATH_NO_TOP,
        cache_subdir='models'
    )
    model.load_weights(weights_path)

    return model
