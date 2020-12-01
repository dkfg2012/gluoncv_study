# [資料分析&機器學習] 第5.1講: 卷積神經網絡介紹

import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() #there are 60000 training data, and 10000 test data

x_train4d = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') #add 1d data refer to color of the picture
x_test4d = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

#draw the pic in dataset
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(3,3)
    plt.imshow(image, cmap='binary')
    plt.show()
# plot_image(x_Train[0])

#draw list of input image with their label and prediction (if prediction is empty, then no prediction)
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25:
        num = 25
    #draw num picture, at most 25
    for i in range(0, num):
        ax = plt.subplot(5,5,i)
        ax.imshow(images[idx], cmap='binary')
        title = 'label=' + str(labels[idx])
        if len(prediction) > 0:
            title+=",predict="+str(prediction[idx])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
# plot_images_labels_prediction(x_train,y_train,[],0,10)

#normalize the data in range 0 to 1
x_train4d_normalize = x_train4d / 255 #all data in dataset divided by 255
x_test4d_normalize = x_test4d / 255

#do one hot encoding
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

#create cnn
model = Sequential()
#add convolution layer
model.add(
    Conv2D(filters=16, kernel_size = (5,5), padding = "same", input_shape=(28, 28, 1), activation = "relu") #we take 5x5 square data each time, 16 feature detectors, use relu to remove all negative value
)

#add max pooling
model.add(
    MaxPooling2D(pool_size = (2,2)) #max pooling by getting 2x2 data
)
model.add(
    Conv2D(filters=36, kernel_size=(5,5), padding='same', activation = 'relu')
)
model.add(MaxPooling2D(pool_size=(2,2)))
#drop to prevent overfitting
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# train model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train_history = model.fit(x=x_train4d_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=20, batch_size=300, verbose=2)

#plot accuracy and loss function graph
# def show_train_history(train_acc,test_acc):
#     plt.plot(train_history.history[train_acc])
#     plt.plot(train_history.history[test_acc])
#     plt.title('Train History')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#
# show_train_history('acc','val_acc')
# show_train_history('loss','val_loss')

#evaluate model accuracy
# scores = model.evaluate(x_test4d_normalize , y_TestOneHot)

#do prediction based on test data set
# prediction=model.predict_classes(x_test4d_normalize)
# plot_images_labels_prediction(x_test,y_test,prediction,idx=0) # we contain a prediction now