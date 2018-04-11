import keras  
from keras.layers import LSTM  
from keras.layers import Dense, Activation, Input, Dropout, Activation
from keras.datasets import mnist  
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

learning_rate = 0.001  
training_iters = 3  
batch_size = 128  
display_step = 10  

n_input = 28  
n_step = 28 
n_hidden = 128  
n_classes = 10  

(x_train, y_train), (x_test, y_test) = mnist.load_data()  
x_train = x_train.reshape(-1, n_step, n_input)  
x_test = x_test.reshape(-1, n_step, n_input)  
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')  
x_train /= 255  
x_test /= 255  

y_train = keras.utils.to_categorical(y_train, n_classes)  
y_test = keras.utils.to_categorical(y_test, n_classes)  


# 模型定义1：序贯模型 Sequential
#model = Sequential()  
#model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), unroll=True)) 
#model.add(Dense(n_classes))  
#model.add(Activation('softmax')) 


# 模型定义2：函数式模型 Model，Sequential是Model的特殊情况
#inputs = Input(shape=(n_step, n_input))
#x = LSTM(n_hidden, unroll=True)(inputs)
#predictions  = Dense(n_classes, activation='softmax')(x)
#model = Model(inputs=inputs, outputs=predictions)

inputs = Input(shape=(n_step, n_input))
X = LSTM(n_hidden, return_sequences=True)(inputs)
X = Dropout(0.5)(X)
X = LSTM(128)(X)
X = Dropout(0.5)(X)
X = Dense(n_classes)(X)
predictions = Activation('softmax')(X)
model = Model(inputs=inputs, outputs=predictions)


adam = Adam(lr=learning_rate)  
model.summary()  
model.compile(optimizer=adam,  
              loss='categorical_crossentropy',  
              metrics=['accuracy'])  

model.fit(x_train, y_train,  
          batch_size=batch_size,  
          epochs=training_iters,  
          verbose=1,  
          validation_data=(x_test, y_test),
          callbacks=[TensorBoard(log_dir='./logs/keras_mnist_lstm/')])  

scores = model.evaluate(x_test, y_test, verbose=0)  
print('LSTM test score:', scores[0])  
print('LSTM test accuracy:', scores[1])