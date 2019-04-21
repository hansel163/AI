import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.client import device_lib
import keras
import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import utils
import matplotlib.pyplot as plt

np.random.seed(10)

def list_local_device():
    print(device_lib.list_local_devices())

def tf_hello_world():
    tfs = tf.InteractiveSession()

    hello = tf.constant("Hello World!")
    print(tfs.run(hello))

def linear_regression():
    w = tf.Variable([0.3], name='w', dtype=tf.float32)
    b = tf.Variable([-0.3], name='b', dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    y = w * x + b
    output = 0
    with tf.Session() as tfs:
        tf.global_variables_initializer().run()
        # tf logs for tensorboard 'tensorboard --logdir=tflogs '
        writer = tf.summary.FileWriter('tflogs', tfs.graph)
        output = tfs.run(y, {x:[1,2,3,4]})
    print('output: ', output)

def test_MINIST():
    # define some hyper parameters
    batch_size = 100
    n_inputs = 784
    n_classes = 10
    n_epochs = 10

    # get the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape the two dimensional 28 x 28 pixels
    #   sized images into a single vector of 784 pixels
    x_train = x_train.reshape(60000, n_inputs)
    x_test = x_test.reshape(10000, n_inputs)

    # convert the input values to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # normalize the values of image vectors to fit under 1
    x_train /= 255
    x_test /= 255

    # convert output data into one hot encoded format
    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)

    # build a sequential model
    model = Sequential()
    # the first layer has to specify the dimensions of the input vector
    model.add(Dense(units=128, activation='sigmoid', input_shape=(n_inputs,)))
    # add dropout layer for preventing overfitting
    model.add(Dropout(0.1))
    model.add(Dense(units=128, activation='sigmoid'))
    model.add(Dropout(0.1))
    # output layer can only have the neurons equal to the number of outputs
    model.add(Dense(units=n_classes, activation='softmax'))

    # print the summary of our model
    model.summary()

    # compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(),
                metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs)

    # evaluate the model and print the accuracy score
    scores = model.evaluate(x_test, y_test)

    print('\n loss:', scores[0])
    print('\n accuracy:', scores[1])

def test_cifar10():
    (x_img_train,y_label_train),(x_img_test, y_label_test)=cifar10.load_data() 

label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer", 5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"} 
def plot_cifar10(images,labels,prediction,idx,num=10): 
    fig = plt.gcf() 
    fig.set_size_inches(12, 14) # 控制图片大小 if num>25: num=25 #最多显示25张 
    for i in range(0, num): 
        ax=plt.subplot(5,5, 1+i) 
        ax.imshow(images[idx],cmap='binary') 
        title=str(i)+','+label_dict[labels[i][0]]# i-th张图片对应的类别 
        if len(prediction)>0: 
            title+='=>'+label_dict[prediction[i]] 
            ax.set_title(title,fontsize=10) 
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            idx+=1 
            plt.savefig('1.png') 
            plt.show()


def main():
    # list_local_device()
    test_MINIST()
    # test_cifar10()

if __name__ == '__main__':
    main()