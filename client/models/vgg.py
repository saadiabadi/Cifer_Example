'''VGG11/13/16/19 in keras.'''
import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
import random

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


trainable_layers = {
    'VGG11': [0, 2, 4, 5, 7, 8, 10, 11],
    'VGG13': [0, 1, 3, 4, 6, 7, 9, 10, 11, 12],
    'VGG16': [0, 1, 3, 4,  6, 7, 8,  10, 11, 12, 14, 15, 16],
    'VGG19': [0, 1,  3, 4,  6, 7, 8, 9,  11, 12, 13, 14,  16, 17, 18, 19],
}


def create_seed_model(input_shape=(32,32,3), dimension='VGG16', trainedLayers=0):

    num_classes = 10
    lay_count = 0

    if trainedLayers > 0:

        randomlist = random.sample(trainable_layers[dimension], trainedLayers)
        print(randomlist)
        model = Sequential()
        model.add(tensorflow.keras.Input(shape=input_shape))
        for x in cfg[dimension]:
            if x == 'M':
                model.add(MaxPooling2D(pool_size=(2, 2)))
            else:
                if lay_count in randomlist:
                    model.add(Conv2D(x, (3, 3), padding='same', trainable=True))
                    model.add(BatchNormalization(trainable=True))
                    model.add(Activation(activations.relu))
                else:
                    model.add(Conv2D(x, (3, 3), padding='same', trainable=False))
                    model.add(BatchNormalization(trainable=False))
                    model.add(Activation(activations.relu))
            lay_count += 1

        #model.add(Flatten())
        model.add(AveragePooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

        print(" --------------------------------------- ")
        print(" ------------------MODEL CREATED------------------ ")
        print(" --------------------------------------- ")

    else:
        model = Sequential()
        model.add(tensorflow.keras.Input(shape=input_shape))
        for x in cfg[dimension]:
            if x == 'M':
                model.add(MaxPooling2D(pool_size=(2, 2)))
            else:
                print("trani: ", x)
                model.add(Conv2D(x, (3, 3), padding='same', trainable=True))
                model.add(BatchNormalization(trainable=True))
                model.add(Activation(activations.relu))

        # model.add(Flatten())
        model.add(AveragePooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])

    return model


### generate a full trainable seed model

# def create_seed_model(input_shape=(32, 32, 3), dimension='VGG16'):
#     num_classes = 10
#     model = Sequential()
#     model.add(keras.Input(shape=input_shape))
#     for x in cfg[dimension]:
#         if x == 'M':
#             model.add(MaxPooling2D(pool_size=(2, 2)))
#         else:
#             print("trani: ", x)
#             model.add(Conv2D(x, (3, 3), padding='same', trainable=True))
#             model.add(BatchNormalization(trainable=True))
#             model.add(Activation(activations.relu))
#
#     # model.add(Flatten())
#     model.add(AveragePooling2D(pool_size=(1, 1)))
#     model.add(Flatten())
#     model.add(Dense(num_classes, activation='softmax'))
#     opt = keras.optimizers.Adam(learning_rate=0.001)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt, metrics=['accuracy'])
#
#     return model



