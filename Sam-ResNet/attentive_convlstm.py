import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv2D, InputSpec
from tensorflow.keras import activations, initializers


class AttentiveConvLSTM(Layer):
    """
    An updated AttentiveConvLSTM that uses modern TensorFlow/Keras 2.x APIs.
    Expects 5D input: (batch, timesteps, channels, height, width).
    """
    def __init__(self,
                 nb_filters_in,
                 nb_filters_out,
                 nb_filters_att,
                 nb_rows,
                 nb_cols,
                 kernel_initializer='glorot_uniform',
                 inner_initializer='orthogonal',
                 attentive_initializer='zeros',
                 activation='tanh',
                 inner_activation='sigmoid',
                 go_backwards=False,
                 **kwargs):
        super(AttentiveConvLSTM, self).__init__(**kwargs)

        # ---- Store hyperparameters ----
        self.nb_filters_in = nb_filters_in
        self.nb_filters_out = nb_filters_out
        self.nb_filters_att = nb_filters_att
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.inner_initializer = initializers.get(inner_initializer)
        self.attentive_initializer = initializers.get(attentive_initializer)

        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.go_backwards = go_backwards

        # We'll have two hidden states: (h, c)
        self.states = [None, None]
        self.input_spec = [InputSpec(ndim=5)]

        # ---- Define sub-layers (Conv2D) ----
        # Attentive parts
        self.W_a = Conv2D(self.nb_filters_att,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.kernel_initializer)

        self.U_a = Conv2D(self.nb_filters_att,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.kernel_initializer)

        self.V_a = Conv2D(1,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=False,
                          kernel_initializer=self.attentive_initializer)

        # i gate
        self.W_i = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.kernel_initializer)

        self.U_i = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.inner_initializer)

        # f gate
        self.W_f = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.kernel_initializer)

        self.U_f = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.inner_initializer)

        # c gate
        self.W_c = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.kernel_initializer)

        self.U_c = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.inner_initializer)

        # o gate
        self.W_o = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.kernel_initializer)

        self.U_o = Conv2D(self.nb_filters_out,
                          kernel_size=(self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer=self.inner_initializer)

    def build(self, input_shape):
        # input_shape should be (batch, timesteps, channels, height, width)
        # Let Keras handle building of sub-layers automatically.
        # We just call self.built = True at the end.
        super(AttentiveConvLSTM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, timesteps, channels, height, width)
        # This returns only the final hidden state per sequence:
        # shape = (batch, nb_filters_out, height, width)
        batch, timesteps, _, height, width = input_shape
        return (batch, self.nb_filters_out, height, width)

    def compute_mask(self, inputs, mask=None):
        # We do not provide a mask for the output
        return None

    def preprocess_input(self, x):
        # For this architecture, we do not need a special preprocess
        return x

    def get_constants(self, x):
        # No extra constants used
        return []

    def step(self, x_t, states):
        """
        One time-step of the AttentiveConvLSTM.
        x_t: shape (batch, channels, height, width)
        states: [h_tm1, c_tm1]
        """
        h_tm1 = states[0]  # (batch, nb_filters_out, height, width)
        c_tm1 = states[1]  # (batch, nb_filters_out, height, width)

        # 1) Compute attention
        # e shape => (batch, 1, height, width)
        e = self.V_a(K.tanh(self.W_a(h_tm1) + self.U_a(x_t)))
        # Flatten => softmax => reshape
        e_reshaped = K.batch_flatten(e)  # (batch, height*width)
        alpha = K.softmax(e_reshaped)    # (batch, height*width)
        # reshape alpha => (batch, 1, height, width)
        alpha = K.reshape(alpha, (K.shape(e)[0], 1, K.shape(e)[2], K.shape(e)[3]))

        # 2) Weighted input x_tilde
        # alpha shape = (b_s, 1, H, W)
        # channels = tf.shape(x_t)[1]

        channels = tf.shape(x_t)[1]
        alpha_tiled = tf.tile(alpha, [1, channels, 1, 1])
        # alpha_tiled shape = (b_s, channels, H, W)

        x_tilde = x_t * alpha_tiled

        # 3) LSTM gates
        # i gate
        x_i = self.W_i(x_tilde)
        i = self.inner_activation(x_i + self.U_i(h_tm1))
        # f gate
        x_f = self.W_f(x_tilde)
        f = self.inner_activation(x_f + self.U_f(h_tm1))
        # c
        x_c = self.W_c(x_tilde)
        c = f * c_tm1 + i * self.activation(x_c + self.U_c(h_tm1))
        # o gate
        x_o = self.W_o(x_tilde)
        o = self.inner_activation(x_o + self.U_o(h_tm1))

        # new hidden
        h = o * self.activation(c)

        return h, [h, c]

    def call(self, x, mask=None):
        """
        x shape = (batch, timesteps, channels, height, width).
        """
        # 1) Extract dynamic shape
        shape = tf.shape(x)  # shape => [batch, timesteps, channels, height, width]
        batch = shape[0]
        timesteps = shape[1]
        channels = shape[2]
        height = shape[3]
        width = shape[4]

        # 2) Initialize states here, so no out-of-scope
        h0 = tf.zeros((batch, self.nb_filters_out, height, width), dtype=x.dtype)
        c0 = tf.zeros((batch, self.nb_filters_out, height, width), dtype=x.dtype)
        initial_states = [h0, c0]

        # 3) Possibly define any “constants” or “preprocessing” if needed
        constants = self.get_constants(x)  # or keep it in call if it’s simple

        # 4) Run your custom RNN or K.rnn(...)
        last_output, outputs, states = K.rnn(
            step_function=self.step,  # or self.step(...),
            inputs=x,  # or preprocessed x
            initial_states=initial_states,
            go_backwards=self.go_backwards,
            mask=mask,
            constants=constants,
            unroll=False,
            input_length=timesteps
        )

        return last_output

