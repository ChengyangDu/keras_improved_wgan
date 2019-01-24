import os
from keras.layers import Dense, Conv2DTranspose, Reshape, UpSampling2D, Conv2D, LeakyReLU, Flatten, Activation, BatchNormalization
from keras.models import Sequential
import numpy as np
import tensorflow as tf

class Net(object):
    def __init__(self, gen_model=None, dis_model=None):
        if gen_model is None:
            # W=(N−1)∗S−2P+F
            gen_model = Sequential()

            # stage one: dense 12*12*512
            gen_model.add(Dense(3*3*8192, activation='relu', input_dim=128))
            gen_model.add(Activation('relu'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Reshape([8192, 3, 3]))


            # 3*3*8192 -> 16*16*64: 7 block: +1+2+2+2+2+2+2
            # stage two: W = N+ F-1
            gen_model.add(Conv2DTranspose(1024, 2, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(1024, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(1024, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(512, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(256, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(128, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(64, 3, data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            

            #16*16*64 -> 512*512*1: 5 block
            # stage three: W = 2*N ('same' == 1)
            gen_model.add(Conv2DTranspose(32, 4, strides=2, padding='same', data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(16, 4, strides=2, padding='same', data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(8, 4, strides=2, padding='same', data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))

            gen_model.add(Conv2DTranspose(4, 4, strides=2, padding='same', data_format='channels_first'))
            gen_model.add(BatchNormalization(axis=1))
            gen_model.add(Activation('relu'))            

            gen_model.add(Conv2DTranspose(1, 4, strides=2, padding='same', data_format='channels_first'))
            gen_model.add(Activation('tanh'))
        self.generator = gen_model



        if dis_model is None:
            # N=(W−F+2P)/S+1
            dis_model = Sequential()

            # stage one: N = (W-5+2*1)/2+1 = (W-3)/2+1 = W/2
            dis_model.add(Conv2D(32, 5, strides=2, padding='same', data_format='channels_first', input_shape=[1, 512, 512]))
            dis_model.add(LeakyReLU(0.2))

            dis_model.add(Conv2D(64, 5, strides=2, padding='same', data_format='channels_first'))
            dis_model.add(LeakyReLU(0.2))

            dis_model.add(Conv2D(128, 5, strides=2, padding='same', data_format='channels_first'))
            dis_model.add(LeakyReLU(0.2))

            # # stage two:
            dis_model.add(Conv2D(256, 5, strides=2, data_format='channels_first'))
            dis_model.add(LeakyReLU(0.2))

            dis_model.add(Conv2D(512, 5, strides=2, data_format='channels_first'))
            dis_model.add(LeakyReLU(0.2))

            dis_model.add(Conv2D(1024, 5, strides=2, data_format='channels_first'))
            dis_model.add(LeakyReLU(0.2))           
            
            dis_model.add(Flatten())
            dis_model.add(Dense(256))

            dis_model.add(LeakyReLU(0.2))
            dis_model.add(Dense(1))
        self.discriminator = dis_model

    def save_models(self, name, save_dir='save'):
        self.generator.save(os.path.join(save_dir, "generator_{}.h5".format(name)))
        self.discriminator.save(os.path.join(save_dir, "discriminator_{}.h5".format(name)))

if __name__ == '__main__':
    net = Net()

