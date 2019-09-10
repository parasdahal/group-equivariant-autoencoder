import os, sys
import numpy as np
import keras
from keras.layers import Conv2D, Layer, MaxPool2D, Dense, Flatten, Activation, Reshape, UpSampling2D, Add, Multiply, Lambda, Input, InputLayer
from keras_gcnn.layers import GConv2D, GBatchNorm
from keras_gcnn.layers.pooling import GroupPool
import keras.backend as K


class ConvAutoEncoder:

  def __init__(self,
               input_shape,
               output_dim,
               use_gconv=True,
               kernel=(3, 3),
               stride=(1, 1),
               strideundo=2,
               pool=(2, 2),
               vae=False):

    self.input_shape = input_shape
    self.output_dim = output_dim
    self.use_gconv = use_gconv
    self.vae = vae

    self.encoder = Encoder(use_gconv,
                           input_shape,
                           output_dim,
                           kernel,
                           stride,
                           pool,
                           vae=vae)
    self.decoder = Decoder(use_gconv,
                           input_shape,
                           output_dim,
                           kernel,
                           stride,
                           pool,
                           vae=vae)

    x = keras.layers.Input(input_shape)
    z = self.encoder(x)
    x_prime = self.decoder(z)

    self.ae = keras.models.Model(inputs=x, outputs=x_prime)
    self.ae.compile(optimizer='adam', loss='binary_crossentropy')

    self.encoder_predict = keras.models.Model(inputs=x, outputs=z).predict

  def model_summary(self):
    print(self.encoder.summary())
    print(self.decoder.summary())

  def fit(self, data, epochs=25, callbacks=[]):

    callbacks = [keras.callbacks.BaseLogger()]

    self.ae.fit(x=data,
                y=data,
                epochs=epochs,
                validation_data=None,
                callbacks=callbacks,
                batch_size=128)

  def save_weights(self, path=None, prefix=""):
    if path is None: path = os.getcwd()
    self.encoder.save_weights(os.path.join(path, prefix + "encoder_weights.h5"))
    self.decoder.save_weights(os.path.join(path, prefix + "decoder_weights.h5"))

  def load_weights(self, path=None, prefix=""):
    if path is None: path = os.getcwd()
    self.encoder.load_weights(os.path.join(path, prefix + "encoder_weights.h5"))
    self.decoder.load_weights(os.path.join(path, prefix + "decoder_weights.h5"))


class Encoder(Layer):

  def __init__(self,
               use_gconv,
               input_shape,
               output_dim,
               kernel,
               stride,
               pool,
               filters=[32, 32, 64, 64],
               scale_factor=2,
               vae=True,
               **kwargs):
    super(Encoder, self).__init__(name='encoder', **kwargs)
    self.use_gconv = use_gconv
    self.output_dim = output_dim
    self.kernel = kernel
    self.stride = stride
    self.pool = pool
    self.filters = filters
    self.scale_factor = scale_factor
    self.vae = vae
    self.encoder_layers = []

    def build(self, input_batch_shape):

      for index, filter in enumerate(self.filters):
        if self.use_gconv:
          h_input = 'Z2' if index == 0 else 'D4'

          gconv_layer = gconv(filters=int(filter / self.scale_factor),
                              kernel_size=self.kernel,
                              strides=self.stride,
                              padding='same',
                              h_input=h_input,
                              h_output='D4',
                              name='gconv_' + str(index))

          self.encoder_layers.append(gconv_layer)
          self.encoder_layers.append(Activation('relu'))

        else:
          self.encoder_layers.append(
              Conv2D(filters=filter,
                     kernel_size=self.kernel,
                     strides=self.stride,
                     padding='same'))
          self.encoder_layers.append(Activation('relu'))

        maxp_layer = MaxPool2D(pool)
        self.encoder_layers.append(maxp_layer)

      if use_gconv:
        self.encoder_layers.append(GroupPool(h_input='D4'))

      self.encoder_layers.append(Flatten())
      self.encoder_layers.append(Dense(256, activation='relu'))
      self.encoder_layers.append(Dense(self.output_dim, name='z_mean'))

      if self.vae:
        self.kld = KLDivergence(beta=50)
        self.sampling = Sampling()
        self.encoder_layers.append(Dense(self.output_dim, name='z_log_var'))

  def call(self, X, training=None):

    output = X
    num_layers = len(self.encoder_layers)
    if self.vae:
      layers = self.encoder_layers[:num_layers - 2]
      for layer in layers:

        output = layer(output)
        print(layer.output_shape)
      z_mean = self.encoder_layers[num_layers - 2](output)
      z_log_var = self.encoder_layers[num_layers - 1](output)

      z_mean, z_log_var = self.kld([z_mean, z_log_var])
      z = self.sampling([z_mean, z_log_var])
      return z

    else:
      for layer in self.encoder_layers:
        output = layer(output)
      return output


class Decoder(Layer):

  def __init__(self,
               use_gconv,
               input_shape,
               output_dim,
               kernel,
               stride,
               pool,
               filters=[32, 32, 64, 64],
               scale_factor=2,
               vae=True,
               **kwargs):
    super(Decoder, self).__init__(**kwargs)

    self.use_gconv = use_gconv
    self.output_dim = output_dim
    self.kernel = kernel
    self.stride = stride
    self.pool = pool
    self.filters = filters
    self.scale_factor = scale_factor
    self.vae = vae
    self.decoder_layers = []

    def build(self, input_batch_size):

      output_size = int(self.input_shape[0] / (2**len(self.filters)))
      fsize = int(self.filters[0] /
                  self.scale_factor) if self.use_gconv else self.filters[0]

      self.decoder_layers.append(
          Dense(256, input_shape=(10,), activation='relu'))

      self.decoder_layers.append(
          Dense(units=output_size * output_size * fsize, activation='relu'))
      self.decoder_layers.append(Reshape((output_size, output_size, fsize)))

      for index, filter in enumerate(self.filters):
        if use_gconv:
          deconv = gconv(filters=int(filter / self.scale_factor),
                         kernel_size=self.kernel,
                         strides=self.stride,
                         padding='same')
        else:
          deconv = Conv2D(filters=filter,
                          kernel_size=self.kernel,
                          strides=self.stride,
                          padding='same')
        self.decoder_layers.append(deconv)
        self.decoder_layers.append(Activation('relu'))
        self.decoder_layers.append(UpSampling2D())

      self.decoder_layers.append(
          Conv2D(filters=self.input_shape[2],
                 kernel_size=self.kernel,
                 strides=self.stride,
                 padding='same'))

  def call(self, X, training=None):
    output = X
    for layer in self.decoder_layers:
      output = layer(output)
    return output


class KLDivergence(Layer):
  """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

  def __init__(self, beta, *args, **kwargs):
    self.beta = beta
    self.is_placeholder = True
    super(KLDivergence, self).__init__(*args, **kwargs)

  def call(self, inputs):
    mu, log_var = inputs
    kl_batch = -.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

    self.add_loss(self.beta * K.mean(kl_batch), inputs=inputs)
    return inputs


class Sampling(Layer):

  def __init__(self, *args, **kwargs):
    self.epsilon = keras.layers.GaussianNoise(stddev=1)
    super(Sampling, self).__init__(*args, **kwargs)

  def call(self, inputs):

    z_mean, z_log_var = inputs
    batch, dim = K.shape(z_mean)[0], K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    return z
