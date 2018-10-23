import numpy as np
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.datasets import fashion_mnist


'''
Load data
'''
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
train_x = np.expand_dims(train_x, -1)
train_y = np.expand_dims(train_y, -1)
test_x = np.expand_dims(test_x, -1)
test_y = np.expand_dims(test_y, -1)

'''
Build model
'''
input = Input(shape=train_x.shape[1:])
x = Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu')(input)
x = MaxPooling2D(padding='same')(x)
x = Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu')(x)
x = MaxPooling2D(padding='same')(x)
x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = Dense(84, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

model = Model(input, output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

'''
Train model
'''
model.fit(train_x, train_y, epochs=1, batch_size=100)

'''
Evaluate model
'''
res = model.evaluate(test_x, test_y, verbose=0)
print(res)
