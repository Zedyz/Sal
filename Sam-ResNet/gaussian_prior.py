import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.utils import get_custom_objects


class LearningPrior(Layer):
    """
    A modernized version of the old Theano-based LearningPrior layer.
    Expects input shape: (batch, channels, height, width), i.e. channels-first.
    Outputs a Gaussian-based prior map for each Gaussian in nb_gaussian.
    """

    def __init__(self,
                 nb_gaussian,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        """
        :param nb_gaussian: Number of Gaussians to learn
        :param kernel_initializer: Initializer for the parameters (mu_x, mu_y, sigma_x, sigma_y).
        :param kernel_regularizer: Regularizer for the parameter vector.
        :param activity_regularizer: Activity regularizer.
        :param kernel_constraint: Constraint for the parameter vector.
        """
        super(LearningPrior, self).__init__(**kwargs)
        self.nb_gaussian = nb_gaussian

        # Modern equivalents of old 'init', 'W_regularizer', etc.
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        # We'll store the parameter vector (mu_x, mu_y, sigma_x, sigma_y) in a single weight
        # shape = (nb_gaussian * 4,)
        self.input_spec = [InputSpec(ndim=4)]  # (batch, channels, height, width)

    def build(self, input_shape):
        # input_shape = (batch_size, channels, height, width)
        param_shape = (self.nb_gaussian * 4,)  # [mu_x, mu_y, sigma_x, sigma_y]*nb_gaussian

        # Create the trainable weight
        self.W = self.add_weight(
            name='learningprior_params',
            shape=param_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )
        super(LearningPrior, self).build(input_shape)  # needed for Keras

    def compute_output_shape(self, input_shape):
        """
        Output shape = (batch_size, nb_gaussian, height, width)
        """
        batch_size = input_shape[0]
        height = input_shape[2]
        width = input_shape[3]
        return (batch_size, self.nb_gaussian, height, width)

    def call(self, x, mask=None):
        """
        x shape = (batch_size, channels, height, width).
        Returns a 4D tensor: (batch_size, nb_gaussian, height, width)
        """
        # Extract parameter vectors
        # each is length self.nb_gaussian
        mu_x = self.W[:self.nb_gaussian]
        mu_y = self.W[self.nb_gaussian : self.nb_gaussian * 2]
        sigma_x = self.W[self.nb_gaussian * 2 : self.nb_gaussian * 3]
        sigma_y = self.W[self.nb_gaussian * 3 : ]

        # Possibly you want these in [0,1], or some clipped range
        mu_x = K.clip(mu_x, 0.25, 0.75)
        mu_y = K.clip(mu_y, 0.35, 0.65)
        sigma_x = K.clip(sigma_x, 0.1, 0.9)
        sigma_y = K.clip(sigma_y, 0.2, 0.8)

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[2]
        width = tf.shape(x)[3]

        # Example aspect ratio logic from old code:
        # e = height/width => e1 = (1-e)/2, e2 = e1 + e
        # If your data is strictly height>=width, watch out for integer division
        # We'll do float division:
        e = tf.cast(height, K.floatx()) / tf.cast(width, K.floatx())
        e1 = (1.0 - e) / 2.0
        e2 = e1 + e

        # Create coordinate grids:
        # x_t: shape (height, width)
        # y_t: shape (height, width)
        # We map x_t in [0, 1], y_t in [e1, e2]
        x_lin = tf.linspace(0.0, 1.0, width)      # shape (width,)
        y_lin = tf.linspace(e1, e2, height)       # shape (height,)

        # Expand & tile to form full 2D grids
        # x_lin => (1, width)
        x_lin_2d = tf.expand_dims(x_lin, axis=0)  # shape (1, width)
        # tile => (height, width)
        x_t = tf.tile(x_lin_2d, [height, 1])      # shape (height, width)

        # y_lin => (height, 1)
        y_lin_2d = tf.expand_dims(y_lin, axis=1)  # shape (height, 1)
        # tile => (height, width)
        y_t = tf.tile(y_lin_2d, [1, width])       # shape (height, width)

        # Expand dims so we can have 'nb_gaussian' channels
        # x_t => (height, width, 1)
        x_t = tf.expand_dims(x_t, axis=-1)
        y_t = tf.expand_dims(y_t, axis=-1)

        # Now repeat them nb_gaussian times along the last axis => (height, width, nb_gaussian)
        x_t = tf.tile(x_t, [1, 1, self.nb_gaussian])
        y_t = tf.tile(y_t, [1, 1, self.nb_gaussian])

        # Convert mu_x, mu_y, sigma_x, sigma_y to shape (1,1,nb_gaussian)
        mu_x_3d = tf.reshape(mu_x, (1, 1, self.nb_gaussian))
        mu_y_3d = tf.reshape(mu_y, (1, 1, self.nb_gaussian))
        sigma_x_3d = tf.reshape(sigma_x, (1, 1, self.nb_gaussian))
        sigma_y_3d = tf.reshape(sigma_y, (1, 1, self.nb_gaussian))

        # Gaussian formula in 2D
        # gauss = 1/(2*pi*sx*sy) * exp( -((x-mux)^2/(2sx^2) + (y-muy)^2/(2sy^2)) )
        eps = K.epsilon()
        norm = (2.0 * np.pi * sigma_x_3d * sigma_y_3d) + eps
        exponent = (
            (x_t - mu_x_3d)**2 / (2.0 * sigma_x_3d**2 + eps)
            + (y_t - mu_y_3d)**2 / (2.0 * sigma_y_3d**2 + eps)
        )
        gaussian = (1.0 / norm) * tf.exp(-exponent)

        # gaussian shape => (height, width, nb_gaussian)
        # Permute to => (nb_gaussian, height, width)
        gaussian = K.permute_dimensions(gaussian, (2, 0, 1))  # (nb_gaussian, height, width)

        # Normalize each Gaussian by its max value, so max=1
        # shape => (nb_gaussian,)
        max_gauss = tf.reduce_max(gaussian, axis=[1,2], keepdims=True)  # shape (nb_gaussian, 1, 1)
        # Avoid division by zero:
        max_gauss = tf.maximum(max_gauss, eps)

        gaussian = gaussian / max_gauss  # scale to [0,1]

        # Expand batch dimension => (1, nb_gaussian, height, width)
        gaussian = tf.expand_dims(gaussian, axis=0)

        # Tile across batch => (batch_size, nb_gaussian, height, width)
        output = tf.tile(gaussian, [batch_size, 1, 1, 1])

        return output

    def get_config(self):
        config = super(LearningPrior, self).get_config()
        config.update({
            'nb_gaussian': self.nb_gaussian,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        })
        return config
