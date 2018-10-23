import numpy as np
from keras.layers import Input, GlobalAveragePooling2D, Add
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.datasets import fashion_mnist


class ResNet34(object):
    '''
    "Deep Residual Learning for Image Recognition"
    Kaiming He et al.
    https://arxiv.org/abs/1512.03385
    '''
    def __init__(self, input_shape, output_dim):
        x = Input(shape=input_shape)
        h = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(h)
        h = self._building_block(h, channel_out=64)
        h = self._building_block(h, channel_out=64)
        h = self._building_block(h, channel_out=64)
        h = Conv2D(128, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=128)
        h = self._building_block(h, channel_out=128)
        h = self._building_block(h, channel_out=128)
        h = self._building_block(h, channel_out=128)
        h = Conv2D(256, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = Conv2D(512, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=512)
        h = self._building_block(h, channel_out=512)
        h = self._building_block(h, channel_out=512)
        h = GlobalAveragePooling2D()(h)
        h = Dense(1000, activation='relu')(h)
        y = Dense(output_dim, activation='softmax')(h)
        self.model = Model(x, y)

    def __call__(self):
        return self.model

    def _building_block(self, x, channel_out=64):
        h = Conv2D(channel_out, kernel_size=(3, 3), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(channel_out, kernel_size=(3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Add()([h, x])
        return Activation('relu')(h)


if __name__ == '__main__':
    '''
    Build model
    '''
    resnet = ResNet34((224, 224, 3), 10)
    model = resnet()
    model.summary()
