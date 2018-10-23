import numpy as np
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.datasets import fashion_mnist


class LeNet(object):
    def __init__(self, input_shape, output_dim):
        x = Input(shape=input_shape)
        h = Conv2D(6, kernel_size=(5, 5),
                   padding='valid', activation='relu')(x)
        h = MaxPooling2D(padding='same')(h)
        h = Conv2D(16, kernel_size=(5, 5),
                   padding='valid', activation='relu')(h)
        h = MaxPooling2D(padding='same')(h)
        h = Flatten()(h)
        h = Dense(120, activation='relu')(h)
        h = Dense(84, activation='relu')(h)
        y = Dense(output_dim, activation='softmax')(h)
        self.model = Model(x, y)

    def __call__(self):
        return self.model


if __name__ == '__main__':
    '''
    Build model
    '''
    lenet = LeNet((30, 30, 1), 10)
    model = lenet()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
