import load_wav_files as lw
import numpy as np

'''import numpy as np
import scipy.io.wavfile as wav

import speech_features.audio_processing as psf

(rate, signal) = wav.read("speech_commands_v0.01/cat/af7a8296_nohash_1.wav")

#feature = mfcc_feature(signal, rate, winfunc = np.hamming, nfil)

# Nel peaper usano queasto
feature, _ = psf.mel_fbank(signal, rate, nfilt = 40, winfunc = np.hamming)
print(feature.shape)'''

X_train = lw.load_dataset('speech_commands_v0.01', {'bed'}, 0.5,  'training')


print(X_train)

#X_test = lw.load_dataset('speech_commands_v0.01', {'bed'}, 'testing')

#print(len(X_train))
#print(len(X_test))
'''
for i in np.arange(1,len(X_train), 2):
    if(np.isscalar(X_train[i]['feature']) or np.isscalar(X_train[i-1]['feature'])):
        continue
    if(len(X_train[i]['feature']) != len(X_train[i-1]['feature'])):
        print(len(X_train[i-1]['feature']))
        print(len(X_train[i]['feature']))

'''
