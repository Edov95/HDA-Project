import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

def convolutional_network_2_layer(n_kws, feats_shape, optimizer = None, learning_rate=0.002):

    model = Sequential()
    model.add(Conv2D(16, 3, strides = (1, 1), input_shape = feats_shape,
                     padding = 'valid', data_format = 'channels_last',
                     dilation_rate = (1, 1), activation = 'relu',
                     use_bias = True, bias_initializer = 'random_uniform',
                     kernel_initializer = 'normal'))

    model.add(MaxPooling2D((2,2) strides = (2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3,3), strides=(1, 1),  padding='valid',
                     data_format='channels_last', dilation_rate=(1, 1),
                     activation='relu', use_bias=True,
                     kernel_initializer='normal',
                     bias_initializer='random_uniform'))
    model.add(MaxPooling2D((2,2) strides = (1, 1)))
    model.add(Conv2D(64, (3,3), strides=(1, 1),  padding='valid',
                     data_format='channels_last', dilation_rate=(1, 1),
                     activation='relu', use_bias=True,
                     kernel_initializer='normal',
                     bias_initializer='random_uniform'))

    model.add(MaxPooling2D((2,2) strides = (1, 1)))                 
    model.add(Flatten())
    model.add(Dense(n_kws, activation='softmax'))

    if optimizer == None:
        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(path_to_saved_files, model, verb, n_epochs, save_model,
                path_to_save_model, batch_size):
    # Carico i dataset già creati
    if verb:
        print('convolutional_network_2_layer: loading the train dataset...')
    x_train = np.load(os.path.join(path_to_saved_files,'x_train.npy'))
    y_train = np.load(os.path.join(path_to_saved_files,'y_train.npy'))
    print('convolutional_network_2_layer: train dataset loaded')

    if verb:
        print('convolutional_network_2_layer: loading the validate dataset...')
    x_validate = np.load(os.path.join(path_to_saved_files,'x_validation.npy'))
    y_validate = np.load(os.path.join(path_to_saved_files,'y_validation.npy'))
    print('convolutional_network_2_layer: validate dataset loaded')

    if verb:
        print('convolutional_network_2_layer: loading the test dataset...')
    x_test = np.load(os.path.join(path_to_saved_files,'x_test.npy'))
    y_test = np.load(os.path.join(path_to_saved_files,'y_test.npy'))
    print('convolutional_network_2_layer: test dataset loaded')

    x_train = x_train / np.max(x_train)
    x_validate = x_validate / np.max(x_validate)
    x_test = x_test / np.max(x_test)

    feat_cols = x_train.shape[1]
    feat_rows = x_train.shape[2]

    feats_shape = (feat_cols, feat_rows, 1)

    x_train = x_train.reshape(x_train.shape[0], *feats_shape)
    x_test = x_test.reshape(x_test.shape[0], *feats_shape)
    x_validate = x_validate.reshape(x_validate.shape[0], *feats_shape)


    if verb:
        print('batch_size: {}'.format(batch_size))
        print('feature row: {}'.format(feat_rows))
        print('feature cols: {}'.format(feat_cols))

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=1,
                                                verbose=1,
                                                factor=0.1,
                                                min_lr=0.00001)

    model.fit(x_train, y_train, batch_size = batch_size, epochs = n_epochs,
              validation_data = (x_validate, y_validate), verbose = verb,
              callbacks = [learning_rate_reduction])

    score = model.evaluate(x_test, y_test, batch_size = batch_size,
                           verbose = verb)

    if verbose:
        print('test loss: {:.4f}'.format(score[0]))
        print('test acc: {:.4f}'.format(score[1]))

    if save_model:
        model.save(os.path.absolute(path_to_save_model))

    return model


'''
x_train, y_train = lw.load_dataset("/nfsd/hda/vaninedoar/HDA-Project/speech_commands_v0.02", {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"}, 0.25, 'training')
print("Loaded train dataset")
x_test, y_test = lw.load_dataset("/nfsd/hda/vaninedoar/HDA-Project/speech_commands_v0.02", {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"}, 0.25, 'testing')
print("Loaded test dataset")
x_validate, y_validate = lw.load_dataset("/nfsd/hda/vaninedoar/HDA-Project/speech_commands_v0.02", {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"}, 0.25, 'validation')
print("Loaded validate dataset")

feat_rows = 99
feat_cols = 40
batch_size = x_train.shape[0]
feats_shape = (99, 40, 1)

x_train = x_train.reshape(x_train.shape[0], *feats_shape)
x_test = x_test.reshape(x_test.shape[0], *feats_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *feats_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))

# Create the CNN
n_kws = 31

#categorical_labels = to_categorical(y_train, num_classes=1)

model = Sequential()
model.add(Conv2D(32, 3, strides = (3, 3), input_shape = feats_shape, padding = 'valid',
                  data_format = 'channels_last', dilation_rate = (1, 1),
                  activation = 'relu', use_bias = True, bias_initializer = 'zeros',
                  kernel_initializer = 'glorot_uniform'))

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), strides=(3, 3),  padding='valid',
                 data_format='channels_last', dilation_rate=(1, 1),
                 activation='relu', use_bias=True, kernel_initializer='random_uniform',
                 bias_initializer='zeros'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(n_kws, activation='softmax'))

#tensorboard = TensorBoard(log_dir = r'\logs{}'.format('cnn_1layer'),
#                         write_graph = True, write_grads = True, histogram_freq = 1, write_images = True)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.002), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs=100, validation_data = (x_validate, y_validate),
          verbose = 1)
score = model.evaluate(x_test, y_test, batch_size=x_test.shape[0], verbose = 0)



print('test loss: {:.4f}'.format(score[0]))
print('test acc: {:.4f}'.format(score[1]))


'''
