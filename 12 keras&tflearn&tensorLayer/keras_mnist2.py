import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from keras import backend as K

from keras.datasets import mnist

import numpy as np


'''http://nooverfit.com/wp/keras-手把手入门1-手写数字识别-深度学习实战/
'''

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 输出
#(60000, 28, 28)
#(60000,)
#(10000, 28, 28)
#(10000,)

batch_size = 128
num_classes = 10
epochs = 10

# 修改数据shape
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


# 模型定义1：序贯模型 Sequential
#model = Sequential()
#model.add(Conv2D(64, activation='relu', input_shape=input_shape, kernel_size = (3,3)))
#model.add(Conv2D(128, activation='relu', kernel_size = (3,3)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.35))
#model.add(Flatten())
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))


# 模型定义2：函数式模型 Model，Sequential是Model的特殊情况
input = Input(shape=input_shape)
x = Conv2D(64, activation='relu', kernel_size = (3,3))(input)
x = Conv2D(128, activation='relu', kernel_size = (3,3))(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.35)(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.35)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=input, outputs=predictions)


model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
