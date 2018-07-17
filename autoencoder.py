import sys
sys.path.insert(0, 'speech_features')

import load_wav_files as lw

import numpy as np

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    #fig, ax = plt.figure()
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    #fig.add_axes(ax)
    plt.savefig('GridSearchCV.png')

def autoencoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #99 x 12 x 32
    pool1 = MaxPooling2D(pool_size=(3, 2))(conv1) #33 x 20 x 32

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1) #33 x 6 x 64
    pool2 = MaxPooling2D(pool_size=(3, 2))(conv2) #11 x 10 x 64

    conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2) #11 x 3 x 128 (small and thick)

    flatten = Flatten()(conv3)
    reshaped = Reshape((11,3,8))(flatten)

    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(reshaped) #11 x 3 x 128
    up1 = UpSampling2D((3,2))(conv4) # 33 x 20 x 128

    conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1) #33 x 6 x 64
    up2 = UpSampling2D((3,2))(conv5) # 99 x 40 x 64

    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv6) # 99 x 12 x 1

    return flatten, decoded

words = {'zero', 'one'}

print("Try to classify {} words".format(len(words)))
print("The words are: {}".format(words))

print("\n")

x_train_clean, y_train = lw.load_dataset("speech_commands_v0.01", words,
    'mfcc', 0, 'training', 1, 0, 10)
print("Clean train dataset loaded")
x_train_noisy, _ = lw.load_dataset("speech_commands_v0.01", words, 'mfcc', 0.25,
    'training', 1, 0, 10)
print("Noisy train dataset loaded")


x_test_clean, y_test = lw.load_dataset("speech_commands_v0.01", words, 'mfcc', 0,
    'testing', 1, 0, 10)
print("Clean test dataset loaded")
x_test_noisy, _ = lw.load_dataset("speech_commands_v0.01", words, 'mfcc',
    0.25, 'testing', 1, 0, 10)
print("Noisy test dataset loaded")


## Reshape dataset
print("Reshaping datasets")
feat_rows = x_train_clean.shape[1]
feat_cols = x_train_clean.shape[2]
feats_shape = (feat_rows, feat_cols, 1)

x_test_clean = x_test_clean / (np.max(x_test_clean) + 1)
x_test_noisy = x_test_noisy / (np.max(x_test_noisy) + 1)
x_train_clean = x_train_clean / (np.max(x_train_clean) + 1)
x_train_noisy = x_train_noisy / (np.max(x_train_noisy) + 1)


x_train_clean = x_train_clean.reshape(x_train_clean.shape[0], *feats_shape)
x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], *feats_shape)
x_test_clean = x_test_clean.reshape(x_test_clean.shape[0], *feats_shape)
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape[0], *feats_shape)

print('\tx_train_clean shape: {}'.format(x_train_clean.shape))
print('\tx_train_noisy shape: {}'.format(x_train_noisy.shape))
print('\tx_test_clean shape: {}'.format(x_test_clean.shape))
print('\tx_test_noisy shape: {}'.format(x_test_noisy.shape))


## Train the autoencoder

print("Autoencoder options:")

batch_size = 128
epochs = 10
inChannel = 1
x, y = x_train_clean.shape[1], x_train_clean.shape[2]
input_img = Input(shape = (x, y, inChannel))

print("\tBatch size: {}".format(batch_size))
print("\tEpochs: {}".format(epochs))

flatten, decoded = autoencoder(input_img)
autoencoder = Model(input_img, decoded)

autoencoder.compile(Adam(lr=0.001), loss='mse')
autoencoder.summary()

autoencoder_train = autoencoder.fit(x_train_noisy, x_train_clean,
    batch_size=batch_size,epochs=3,verbose=1)

score_clean = autoencoder.evaluate(x_test_clean, x_test_clean,
    batch_size = batch_size, verbose = 1)
score_noisy = autoencoder.evaluate(x_test_noisy, x_test_noisy,
    batch_size = batch_size, verbose = 1)

print("\tLoss clean: {}".format(score_clean))
print("\tLoss noisy: {}".format(score_noisy))

## Encoder
enc = Model(input_img, flatten)
print("Encoder created")

encoded_feature = enc.predict(x_train_noisy)
print("\tNoisy train dataset encoded with shape ", encoded_feature.shape)

encoded_test_clean = enc.predict(x_test_clean)
print("\tClean test dataset encoded with shape ", encoded_test_clean.shape)

encoded_test_noisy = enc.predict(x_test_noisy)
print("\tClean test dataset encoded with shape ", encoded_test_noisy.shape)

#SVM
print("SVM classification")

parameters = {'degree': [4, 5], 'C': [10, 11] }
svc = svm.SVC(decision_function_shape = 'ovr', tol = 0.00001, verbose = 1, kernel = 'poly', coef0 = 1)
clf = GridSearchCV(svc, parameters, n_jobs = 4)
clf.fit(encoded_feature, y_train)

plot_grid_search(clf.cv_results_, [4,6], [10, 11], "Degree", "C")

encoded_test_clean_score = clf.score(encoded_test_clean, y_test)
print("Clean test score: ", encoded_test_clean_score)
encoded_test_noisy_noisy = clf.score(encoded_test_noisy, y_test)
print("Clean test score: ", encoded_test_noisy_noisy)

print("Finish!")
