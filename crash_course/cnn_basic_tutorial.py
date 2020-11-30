import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #there are 60000 training data, and 10000 test data

x_train4d = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') #add 1d data refer to color of the picture
x_test4d = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train4d_normalize = x_train4d / 255
x_test4d_normalize = x_test4d / 255

y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = Sequential()
model.add(
    Conv2D(filters=16, kernel_size = (5,5), padding = "same", input_shape=(28, 28, 1), activation = "relu")
)
model.add(
    MaxPooling2D(pool_size = (2,2))
)
model.add(
    Conv2D(filters=36, kernel_size=(5,5), padding='same', activation = 'relu')
)
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=x_train4d_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=20, batch_size=300, verbose=2)


import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')