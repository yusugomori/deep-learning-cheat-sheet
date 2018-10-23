import numpy as np
from keras.layers import Input, GlobalAveragePooling2D, Add
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.datasets import fashion_mnist


class ResNet34(object):
    def __init__(self, input_shape, output_dim):
        x = Input(shape=input_shape)
        h = Conv2D(64, kernel_size=(7, 7), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(h)
        h = self._add_base_block(h, channel_out=64)
        h = self._add_base_block(h, channel_out=128)
        h = self._add_base_block(h, channel_out=256)
        h = self._add_base_block(h, channel_out=512)
        h = GlobalAveragePooling2D()(h)
        h = Dense(1000, activation='relu')(h)
        y = Dense(output_dim, activation='softmax')(h)
        self.model = Model(x, y)

    def __call__(self):
        return self.model

    def _add_base_block(self, x, channel_out=64):
        h = Conv2D(channel_out, kernel_size=(1, 1), strides=(2, 2))(x)
        return self._base_block(h, channel_out=channel_out)

    def _base_block(self, x, channel_out=64):
        h = Conv2D(channel_out, kernel_size=(3, 3), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(channel_out, kernel_size=(3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        shortcut = self._shortcut(x, output_shape=h.get_shape())
        h = Add()([h, shortcut])
        return Activation('relu')(h)

    def _projection(self, x, channel_out):
        return Conv2D(channel_out, kernel_size=(1, 1), padding='same')(x)

    def _shortcut(self, x, output_shape):
        input_shape = x.get_shape()
        channel_in = input_shape[-1]
        channel_out = output_shape[-1]

        if channel_in != channel_out:
            return self._projection(x, channel_out)
        else:
            return x


if __name__ == '__main__':
    '''
    Build model
    '''
    resnet = ResNet34((224, 224, 3), 10)
    model = resnet()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
