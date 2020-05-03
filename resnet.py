
from __future__ import print_function
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras import backend as K
from keras.models import load_model
import torch
from keras.datasets import mnist
import matplotlib.pyplot as plt


# load MNIST data set
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][pixels]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')


# normalization of train and test data

x_train = x_train / 255.0

x_test = x_test / 255.0

# one-hot

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

# parameters/metrics

EPOCHS = 10

INIT_LR = 5e-4

BS = 32

CLASS_NUM = 10

norm_size = 28

# start to train model

print('start to train model')
# define lenet model
seed = 7
np.random.seed(seed)


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):

    if name is not None:

        bn_name = name + '_bn'

        conv_name = name + '_conv'

    else:

        bn_name = None

        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)

    x = BatchNormalization(axis=3, name=bn_name)(x)

    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):

    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')

    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')

    if with_conv_shortcut:

        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)

        x = add([x, shortcut])

        return x

    else:

        x = add([x, inpt])

        return x
def build(width, height, depth, NB_CLASS):
    inpt = Input(shape=(height, width, depth))

    x = ZeroPadding2D((3, 3))(inpt)

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # (56,56,64)

    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

    # (28,28,128)

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

    # (14,14,256)

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    # (7,7,512)

    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))

    # x = AveragePooling2D(pool_size=(7, 7))(x)

    x = AveragePooling2D(pool_size=(1, 1))(x)

    x = Flatten()(x)

    x = Dense(NB_CLASS, activation='softmax')(x)

    # Create a Keras Model

    model = Model(inputs=inpt, outputs=x)

    print(model.summary())

    # Save a PNG of the Model Build

    #plot_model(model, to_file='C:/Users/user/Desktop/queen mary/deep learning/model/img/Resnet34.png')

    # return the constructed network architecture

    return model


model = build(width=norm_size, height=norm_size, depth=1, NB_CLASS=CLASS_NUM)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Use generators to save memory

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,

                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,

                             horizontal_flip=True, fill_mode="nearest")


x_train1=x_train[0:10][0:10]

y_train1=y_train[0:10][0:10]

H = model.fit_generator(aug.flow(x_train1, y_train1, batch_size=BS),

                            steps_per_epoch=len(x_train) // BS,

                            epochs=EPOCHS, verbose=2)



# save model

# method one
model_save_name = 'm_lenet.h5'
#path = F"/content/gdrive/My Drive/{model_save_name}"
#torch.save(model.state_dict(), path)
#model.load_state_dict(torch.load(path))
# save weights
#model.save_weights('/content/gdrive/My Drive/resnet_34.h5')
#model_file = drive.CreateFile({'title' : 'model.h5'})
#model_file.SetContentFile('model.h5')
#model_file.Upload()
#drive.CreateFile({'id': model_file.get('id')})


# load model


# model.load('../h5/m_lenet.h5')







# plot the iteration process

N = EPOCHS

plt.figure()

plt.plot(np.arange(0,N),H.history['loss'],label='loss')

plt.plot(np.arange(0,N),H.history['accuracy'],label='train_acc')

plt.title('Training Loss and Accuracy on mnist-img classifier')

plt.xlabel('Epoch')

plt.ylabel('Loss/Accuracy')

plt.legend(loc='lower left')

#plt.savefig('../figure/Figure_2.png')

tr_loss, tr_accurary = model.evaluate(x_train, y_train)

# tr_loss = 0.039, tr_accurary = 0.98845

# test

te_loss, te_accurary = model.evaluate(x_test, y_test)
# te_loss = 0.042, te_accurary = 0.9861


test_output_x=x_test[99]
test_output_y=y_test[99]
# Calculating loss and accuracy

predictions=model.predict(x_test,verbose=0)
def test_accuracy():

    err = []

    t = 0

    for i in range(predictions.shape[0]):

        if (np.argmax(predictions[i]) == y_test[i]):

            t = t+ 1

        else:

            err.append(i)

    return t, float(t) * 100 / predictions.shape[0], err
p=test_accuracy()
for i in range(5):
    ax1 = fig1.add_subplot(1, 5, i + 1)

    ax1.imshow(x_test[p[2][i]], interpolation='none', cmap=plt.cm.gray)

    ax2 = fig1.add_subplot(2, 5, i + 6)

    ax2.imshow(x_test[p[2][i + 6]], interpolation='none', cmap=plt.cm.gray)

plt.show()

print("True:          {}".format(y_test[p[2][0:5]]))

print("classified as: {}".format(np.argmax(predictions[p[2][0:5]], axis=1)))

print("True:          {}".format(y_test[p[2][6:11]]))

print("classified as: {}".format(np.argmax(predictions[p[2][6:11]], axis=1)))
print(te_loss)
print(te_accurary)