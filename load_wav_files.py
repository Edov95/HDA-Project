import random
import numpy as np
import scipy.io.wavfile as wav
import os
import re
import hashlib
import math

import speech_features.audio_processing as psf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

random.seed(1181349 + 1179018)

def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result

def load_wav(wav_file):
    (rate, signal) = wav.read(wav_file)
    return rate, signal


def load_dataset(data_dir, word_list, noise_percentage, dataset):
    silence_percentage = 0.2
    if dataset == 'training':
        return load_train_dataset(data_dir, word_list, silence_percentage, noise_percentage)
    elif dataset == 'testing':
        return load_test_dataset(data_dir, word_list)

def load_train_dataset(data_dir, word_list, silence_percentage, noise_percentage):
    """ Carico il data set e lo salvo dopo avelo modificato un po' perché
    altrimenti è troppo bello così lo sporco un po'

    Ogni dato a viene caricato"""
    validation_percentage, testing_percentage = 0, 0.1
    X_train = []
    #wav_lists = os.path.join(data_dir, *, '*.wav')
    for word_l in word_list:
        #wav_word_list = os.path.join(data_dir, word_l)
        wav_list = os.path.join(data_dir, word_l, '*.wav')
        for file in gfile.Glob(wav_list):
            _, word = os.path.split(os.path.dirname(file))
            word = word.lower()

            if which_set(file, validation_percentage, testing_percentage) == 'training':
                rate, signal = load_wav(file);
                signal_and_noise = add_noise(signal, rate, 1, os.path.join(data_dir,'_background_noise_'), noise_percentage)
                feature, _ = psf.mel_fbank(signal_and_noise, rate, nfilt = 40, winfunc = np.hamming)
                #if feature.shape[0] != 99:
                #    print(str(len(signal)) + "                 " + str(rate))
                X_train.append({'feature': feature, 'label': word_l})

    # hotspot
    #silence = len(X_train) * silence_percentage
    silence = int(math.ceil(len(X_train) * silence_percentage / 100))
    for _ in range(silence):
        X_train.append({'feature': 0, 'label': "_silence_"})

    random.shuffle(X_train)

    return X_train

def load_test_dataset(data_dir, word_list):
    searchfile = open(os.path.join(data_dir,"testing_list.txt"), "r")
    X_test = []
    for word in word_list:
        for line in searchfile:
            if word in line:
                rate, signal = load_wav(os.path.join(data_dir,line[:-1]))
                feature, _ = psf.mel_fbank(signal, rate, nfilt = 40, winfunc = np.hamming)
                X_test.append({'feature': feature, 'label': word})

    searchfile.close()

    return X_test

def add_noise(signal, rate, len_sec, noise_dir, noise_percentage):
    noise_index = random.randrange(6)
    noise_list = os.path.join(noise_dir,'*.wav')
    noise_filename = gfile.Glob(noise_list)[noise_index]
    _, noise = load_wav(noise_filename)
    rand_stop = len(noise) - rate * len_sec
    initial_offset = random.randrange(rand_stop)
    noise = noise[initial_offset:initial_offset + rate * len_sec]
    signal = np.pad(signal,(0, rate * len_sec - len(signal)), 'constant', constant_values = 0)
    if random.random() < noise_percentage:
        signal_and_noise = np.add(signal,noise)
    else:
        signal_and_noise = signal
    return signal_and_noise

def add_time_shift():
    return
