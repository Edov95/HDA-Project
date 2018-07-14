import numpy as np
from sklearn import svm
import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, Flatten, Dense, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

x_validateMFCC = np.load('compressed_feature/x_validate.npy')
y_validateMFCC = np.load('compressed_feature/y_validate.npy')
x_testMFCC = np.load('compressed_feature/x_test.npy')
y_testMFCC = np.load('compressed_feature/y_test.npy')

x_trainMFCC = np.load('compressed_feature/x_training.npy', allow_pickle = False)
y_trainMFCC = np.load('compressed_feature/y_training.npy', allow_pickle = False)

# da ricordarsi di scalare bene test e validation
x_trainMFCC = x_trainMFCC/np.max(x_trainMFCC)
x_testMFCC = x_testMFCC/np.max(x_testMFCC)
x_validateMFCC = x_validateMFCC/np.max(x_validateMFCC)


print('x_trainMFCC shape: {}'.format(x_trainMFCC.shape))
print('x_testMFCC shape: {}'.format(x_testMFCC.shape))
print('x_validateMFCC shape: {}'.format(x_validateMFCC.shape))

feats_shape = (99, 40, 1)

x_trainMFCC = x_trainMFCC.reshape(x_trainMFCC.shape[0], *feats_shape)
x_testMFCC = x_testMFCC.reshape(x_testMFCC.shape[0], *feats_shape)
x_validateMFCC = x_validateMFCC.reshape(x_validateMFCC.shape[0], *feats_shape)


print('x_trainMFCC shape: {}'.format(x_trainMFCC.shape))
print('x_testMFCC shape: {}'.format(x_testMFCC.shape))
print('x_validateMFCC shape: {}'.format(x_validateMFCC.shape))

# Autoencoder

batch_size = 64
epochs = 5
inChannel = 1
x, y = 99, 40
input_img = Input(shape = (x, y, inChannel))

def encoder(input_img):
    #encoder
    #input = 99 x 40 x 1 (wide and thin)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img) #99 x 40 x 32
    pool1 = MaxPooling2D(pool_size=(3, 2))(conv1) #33 x 20 x 32
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1) #33 x 20 x 64
    pool2 = MaxPooling2D(pool_size=(3, 2))(conv2) #11 x 10 x 64
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2) #11 x 10 x 128 (small and thick)
    flatten = Flatten()(conv3)
    #feature = Dense(50, activation = 'sigmoid')(flatten)
    square = Reshape((33,6,16))(flatten)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(square) #11 x 10 x 128
    up1 = UpSampling2D((3,2))(conv4) # 33 x 20 x 128
    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1) #33 x 20 x 64
    up2 = UpSampling2D((3,2))(conv5) # 99 x 40 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 99 x 40 x 1
    return flatten, decoded

flatten, decoded = encoder(input_img)

autoencoder = Model(input_img, decoded)
autoencoder.compile(Adam(lr=0.04), loss = 'mean_squared_error', metrics = ['categorical_accuracy'])

autoencoder.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=4,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

autoencoder_train = autoencoder.fit(x_trainMFCC, x_trainMFCC, batch_size=batch_size,
    epochs= 1000, verbose=1,validation_data=(x_validateMFCC, x_validateMFCC)
    callbacks = [learning_rate_reduction])

enc = Model(input_img, flatten)
encoded_feature = enc.predict(x_trainMFCC)


clf = svm.SVC(decision_function_shape='ovr')
clf.fit(encoded_feature, y_trainMFCC)

encoded_test = enc.predict(x_testMFCC)

y_pred = clf.predict(encoded_test)

cnt = 0.0
for i in range(y_test.shape[0]):
    if(y_test[i] != y_pred[i]):
        cnt += 1

perc_errors = cnt / y_test.shape[0]
print(perc_errors)
