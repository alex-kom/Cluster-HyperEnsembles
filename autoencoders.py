import numpy as np

from keras import Model
from keras.layers import (Input, Dense, Conv2D, Conv2DTranspose,
                          Flatten, Reshape, Cropping2D)

class Autoencoder:
    '''
    Autoencoder model with dense hidden layers.

    Arguments:
        layer_dims : array-like of int
            The size of the dense layers. The last value is the dimensionality
            of the encoder.
        activation : str
            Activation of the dense layers except the last one.
        output_activation : str
            Activation of the output layer of the decoder.
        initializer : str
            The initializer for the hidden layers.
        batch_size : int
            The batch size for training the model.
        epochs : int
            The number of epochs for training the model.
        loss : str
            The loss for training the model.
        optimizer : str
            The optimizer for training the model.
    '''

    def __init__(self,
                 layer_dims=[32, 16],
                 activation='relu',
                 output_activation='linear',
                 initializer='glorot_uniform',
                 batch_size=32,
                 epochs=100,
                 loss='mse',
                 optimizer='adam'):
        self.layer_dims = layer_dims
        self.activation = activation
        self.output_activation = output_activation
        self.initializer = initializer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer

    def _build(self):
        self._build_encoder()
        self._build_decoder()

        x = Input(shape=(self.input_dim,))
        z = self.encoder(x)
        y = self.decoder(z)
        self.model = Model(inputs=[x], outputs=[y])

    def _build_encoder(self):
        x = Input(shape=(self.input_dim,))
        z = x
        for layer_dim in self.layer_dims[:-1]:
            z = Dense(layer_dim,
                      kernel_initializer=self.initializer,
                      activation=self.activation)(z)
        z = Dense(self.layer_dims[-1], activation='linear')(z)
        self.encoder = Model(inputs=[x], outputs=[z])

    def _build_decoder(self):
        z = Input(shape=(self.layer_dims[-1],))
        y = z
        for layer_dim in reversed(self.layer_dims[:-1]):
            y = Dense(layer_dim,
                      kernel_initializer=self.initializer,
                      activation=self.activation)(y)
        y = Dense(self.input_dim, activation=self.output_activation)(y)
        self.decoder = Model(inputs=[z], outputs=[y])

    def fit(self, X):
        '''
        Fit an autoencoder model on data.

        Arguments:
            X : ndarray of shape (n_observations, n_features)
                The data to fit the model.
        '''
        self.input_dim = X.shape[1]
        self._build()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.fit(X, X,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def encode(self, X):
        '''
        Encode data with the fitted model.

        Arguments:
            X : ndarray of shape (n_observations, n_features)
                The data to be encoded.

        Returns:
            X_encoded : ndarray of (n_observations, h_dim)
                The encoded data.
        '''
        return self.encoder.predict(X)

    def decode(self, X):
        '''
        Decode data with the fitted model.

        Arguments:
            X : ndarray of shape (n_observations, h_dim)
                The data to be decoded.

        Returns:
            X_decoded : ndarray of (n_observations, n_features)
                The decoded data.
        '''
        return self.decoder.predict(X)

    def reconstruct(self, X):
        '''
        Encode and decode data with the fitted model.

        Arguments:
            X : ndarray of shape (n_observations, n_features)
                The data to be reconstructed.

        Returns:
            X_rec : ndarray of (n_observations, n_features)
                The reconstructed data.
        '''
        return self.model.predict(X)


class ConvolutionalAutoencoder:
    '''
    Convolutional autoencoder model.

    Arguments:
        kernel_shapes : List of tuples of ints.
            The shapes of the convolution layer kernels. Each shape is of the
            form (height, width, number of filters).
        h_dim : int
            The dimensionality of the encoder.
        strides: int or array-like of ints.
            The stride for each convolution layer.
        initializer : str
            Initializer of the convolution layers.
        activation : str
            Activation of the convolution layers.
        output_activation : str
            Activation of the output layer of the decoder.
        batch_size : int
            The batch size for training the model.
        epochs : int
            The number of epochs for training the model.
        loss : str
            The loss for training the model.
        optimizer : str
            The optimizer for training the model.
    '''

    def __init__(self,
                 kernel_shapes=[(5, 5, 32), (5, 5, 64)],
                 h_dim=16,
                 strides=2,
                 activation='elu',
                 output_activation='sigmoid',
                 initializer='glorot_uniform',
                 batch_size=32,
                 epochs=100,
                 loss='mse',
                 optimizer='adam'):
        self.kernel_shapes = kernel_shapes
        self.h_dim = h_dim
        self.activation = activation
        self.output_activation = output_activation
        self.initializer = initializer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        if isinstance(strides, int):
            self.strides = [strides] * len(self.kernel_shapes)
        else:
            self.strides = strides
        assert len(self.strides) == len(self.kernel_shapes)

    def _build(self):
        self._infer_shapes()
        self._build_encoder()
        self._build_decoder()

        x = Input(shape=self.img_shape)
        z = self.encoder(x)
        y = self.decoder(z)
        self.model = Model(inputs=[x], outputs=[y])

    def _build_encoder(self):
        x = Input(shape=self.img_shape)
        z = x
        for i, (h, w, k) in enumerate(self.kernel_shapes):
            z = Conv2D(k, (h, w),
                       strides=self.strides[i],
                       padding='same',
                       kernel_initializer=self.initializer,
                       activation=self.activation)(z)
        z = Flatten()(z)
        z = Dense(self.h_dim)(z)

        self.encoder = Model(inputs=[x], outputs=[z])

    def _build_decoder(self):
        z = Input(shape=(self.h_dim,))

        last_shape = self.shapes[-1] + (self.kernel_shapes[-1][-1],)
        y = Dense(np.product(last_shape))(z)
        y = Reshape(last_shape)(y)

        for i in reversed(range(len(self.kernel_shapes))):
            h, w = self.kernel_shapes[i][:2]
            k = self.kernel_shapes[i-1][-1] if i else self.img_shape[-1]
            activation = self.activation if i else self.output_activation
            y = Conv2DTranspose(k, (h, w),
                                strides=self.strides[i],
                                padding='same',
                                kernel_initializer=self.initializer,
                                activation=activation)(y)
            y = self._crop(i, y)

        self.decoder = Model(inputs=[z], outputs=[y])

    def _crop(self, i, y):
        h, w = self.shapes[i+1]
        target_h, target_w = self.shapes[i]
        crop_h = self.strides[i] * h - target_h
        crop_w = self.strides[i] * w - target_w
        if crop_h or crop_w:
            y = Cropping2D(cropping=((0, crop_h), (0, crop_w)))(y)
        return y

    def _infer_shapes(self):
        h, w, _ = self.img_shape
        self.shapes = [(h, w)]
        for stride in self.strides:
            h = int(np.ceil(h / stride))
            w = int(np.ceil(w / stride))
            self.shapes.append((h, w))

    def fit(self, X):
        '''
        Fit a convolutional autoencoder on data.

        Arguments:
            X : ndarray of shape (n_observations, height, width, channels)
                The data to fit the model.
        '''
        self.img_shape = X.shape[1:]
        self._build()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model.fit(X, X,
                       batch_size=self.batch_size,
                       epochs=self.epochs)

    def encode(self, X):
        '''
        Encode data with the fitted model.

        Arguments:
            X : ndarray of shape (n_observations, height, width, channels)
                The data to be encoded.

        Returns:
            X_encoded : ndarray of (n_observations, h_dim)
                The encoded data.
        '''
        return self.encoder.predict(X)

    def decode(self, X):
        '''
        Decode data with the fitted model.

        Arguments:
            X : ndarray of shape (n_observations, h_dim)
                The data to be decoded.

        Returns:
            X_decoded : ndarray of (n_observations, height, width, channels)
                The decoded data.
        '''
        return self.decoder.predict(X)

    def reconstruct(self, X):
        '''
        Encode and decode data with the fitted model.

        Arguments:
            X : ndarray of shape (n_observations, height, width, channels)
                The data to be reconstructed.

        Returns:
            X_rec : ndarray of (n_observations, height, width, channels)
                The reconstructed data.
        '''
        return self.model.predict(X)
