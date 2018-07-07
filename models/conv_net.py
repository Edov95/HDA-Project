import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import sys
import os
sys.path.append(os.path.abspath("/home/edoardo/Progetti/HDA"))
import load_wav_files as lw
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

x_train, y_train = lw.load_dataset("speech_commands_v0.01", {'bed'}, 0.25, 'training')
print("Loaded train dataset")
x_test, y_test = lw.load_dataset("speech_commands_v0.01", {'bed'}, 0.25, 'testing')
print("Loaded test dataset")
x_validate, y_validate = lw.load_dataset("speech_commands_v0.01", {'bed'}, 0.25, 'validation')
print("Loaded validate dataset")

feat_rows = 99
feat_cols = 40
batch_size = 1716
feats_shape = (99, 40, 1)

x_train = x_train.reshape(x_train.shape[0], *feats_shape)
x_test = x_test.reshape(x_test.shape[0], *feats_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *feats_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))

# Create the CNN
n_kws = 31

model = Sequential()
model.add(Conv2D(32, 3, strides = (1, 1), input_shape = feats_shape, padding = 'valid',
                  data_format = 'channels_last', dilation_rate = (1, 1),
                  activation = 'relu', use_bias = True, bias_initializer = 'zeros',
                  kernel_initializer = 'glorot_uniform'))

model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), strides=(1, 1),  padding='valid',
                 data_format='channels_last', dilation_rate=(1, 1),
                 activation='relu', use_bias=True, kernel_initializer='random_uniform',
                 bias_initializer='zeros'))
model.add(MaxPooling2D(1))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(n_kws, activation='softmax'))

tensorboard = TensorBoard(log_dir = r'\logs{}'.format('cnn_1layer'),
                         write_graph = True, write_grads = True, histogram_freq = 1, write_images = True)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.002), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs=15, validation_data = (x_validate, y_validate),
          verbose = 1)
score = model.evaluate(x_test, y_test, batch_size=16, verbose = 0)

print('test loss: {:.4f}'.format(score[0]))
print('test acc: {:.4f}'.format(score[1]))
