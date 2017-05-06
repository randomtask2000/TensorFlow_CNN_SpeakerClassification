import scipy.io.wavfile as wav
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from skimage.util.shape import view_as_windows
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
import librosa
import librosa.display
from scipy import misc
import pandas as pd
from matplotlib.pyplot import specgram
import math
import matplotlib.image as mpimg

"""
Only set this to True if you have the original files. Original files not included in this repo
Download links for original files are in the README
"""

create_data = False 
def import_data():
    #import the data
    #x, fs = librosa.load('data/voice_01.wav', sr = 48000)
    voice_01 = wav.read('data/voice_01_carl_manchester.wav')
    voice_02 = wav.read('data/voice_02_sam_stinson.wav')
    voice_03 = wav.read('data/voice_03_bill_boerst.wav')
    voice_04 = wav.read('data/voice_04_kandice_stehlik.wav')
    voice_05 = wav.read('data/voice_05_julia_niedermaier.wav')
    voice_06 = wav.read('data/voice_06_inah_derby.wav')
    voice_07 = wav.read('data/voice_07_gabriela_cowan.wav')
    voice_08 = wav.read('data/voice_08_tara_dow.wav')
    voice_09 = wav.read('data/voice_09_don_jenkins.wav')
    voice_10 = wav.read('data/voice_10_john_lieder.wav')
    
    #inspect the data
    voices = [voice_01, voice_02, voice_03, voice_04, voice_05, voice_06, voice_07, voice_08, voice_09, voice_10]
    for voice_file in voices:
        print('Sample Rate:', voice_file[0], ':::', 'Shape of Array:', voice_file[1].shape) #2 channel audio
        
    #cut the data to be 3 minutes in length
    #3 minutes is 180 seconds
    target_length = 180
    
    voices_short = []
    for voice_file in voices:
        samples_per_second = voice_file[0]
        target_samples = samples_per_second * target_length
        new_voice = np.delete(voice_file[1], np.s_[target_samples::], 0)
        voices_short.append(new_voice)
    
    print("Saving")
    print("")
    #save the data as a new .wav
    #librosa.output.write_wav('./librosa_test_audio.wav', x, sr = fs)
    wav.write('data/voice_01.wav', rate = 48000, data = voices_short[0])
    wav.write('data/voice_02.wav', rate = 48000, data = voices_short[1])
    wav.write('data/voice_03.wav', rate = 48000, data = voices_short[2])
    wav.write('data/voice_04.wav', rate = 48000, data = voices_short[3])
    wav.write('data/voice_05.wav', rate = 48000, data = voices_short[4])
    wav.write('data/voice_06.wav', rate = 48000, data = voices_short[5])
    wav.write('data/voice_07.wav', rate = 48000, data = voices_short[6])
    wav.write('data/voice_08.wav', rate = 48000, data = voices_short[7])
    wav.write('data/voice_09.wav', rate = 48000, data = voices_short[8])
    wav.write('data/voice_10.wav', rate = 48000, data = voices_short[9])
    print("Finished Saving")
    
if create_data == True:
    import_data()
    
"""
before loading the data, make sure that you have extracted all of the zip files from their folders
"""

#load the .wav data
print("Loading...")
print("")
voice_00 = librosa.load('data/voice_01.wav', sr = 48000)
voice_01 = librosa.load('data/voice_02.wav', sr = 48000)
voice_02 = librosa.load('data/voice_03.wav', sr = 48000)
voice_03 = librosa.load('data/voice_04.wav', sr = 48000)
voice_04 = librosa.load('data/voice_05.wav', sr = 48000)
voice_05 = librosa.load('data/voice_06.wav', sr = 48000)
voice_06 = librosa.load('data/voice_07.wav', sr = 48000)
voice_07 = librosa.load('data/voice_08.wav', sr = 48000)
voice_08 = librosa.load('data/voice_09.wav', sr = 48000)
voice_09 = librosa.load('data/voice_10.wav', sr = 48000)
print("Loaded")
print("")

#inspect the data
voices = [voice_00, voice_01, voice_02, voice_03, voice_04, voice_05, voice_06, voice_07, voice_08, voice_09]
for voice_file in voices:
    print('Sample Rate:', voice_file[1], '|', 'Shape of Array:', voice_file[0].shape) #2 channel audio

#only use the audio data (librosa load function also outputs sample rate)
all_voice_files = []
for voice in voices:
    all_voice_files.append(voice[0])
    
#look at the attributes of each file
#min, max, mean, standard deviation
indexer = 0
for voice in all_voice_files:
    print("Sample", indexer, "|", "Mean:", np.mean(voice), "|", "Max:", np.max(voice), "|", \
          "Min:", np.min(voice), "|", "Std Dev:", np.std(voice))
    indexer += 1
    
"""
Find all 200ms samples that have clear audio data in them
"""
#in each sample, find 200ms chunks that have a mean over the total average + standard deviation
window_shape = 48000 / 5

voice_data = []
voice_labels = []
voice_number = 0

for voice in all_voice_files:
    positive_full_array = view_as_windows(np.absolute(voice), window_shape)
    positive_full_array = positive_full_array[::int(window_shape / 2)] #keep every nth row, where n is window_shape/2 (For some overlap)
    temp_full_array = view_as_windows(voice, window_shape)
    temp_full_array = temp_full_array[::int(window_shape / 2)]
    
    for window_index in range(len(temp_full_array)):
        if np.mean(positive_full_array[window_index]) > (np.mean(voice) + np.std(voice)):
            voice_data.append(temp_full_array[window_index])
            voice_labels.append(voice_number)
            
    voice_number += 1
    
voice_data = np.array(voice_data)
print("Number of samples:", voice_data.shape)
#normalize the data
#voice_data_normalized = preprocessing.normalize(voice_data)

voice_labels = np.array(voice_labels)
#one-hot encode the labels
voice_labels = np.eye(10)[voice_labels]

print("Number of labels:", voice_labels.shape)

#determine how many of each speaker is in the samples/labels dataset
identity_matrix = np.identity(10)
number_of_samples = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(voice_labels)):
    for j in range(10):
        number_of_samples[j] += np.sum(np.all(np.equal(voice_labels[i], identity_matrix[j])))

for i in range(10):
    print("Number of samples from voice", i, ':', number_of_samples[i])

"""
split the data into sets
"""

#split into training and testing sets - 60% train, 20% validation, 20% test
X_train, X_test, y_train, y_test = train_test_split(voice_data, voice_labels, test_size = 0.40, random_state = 7)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 7)
print("Training Set:", X_train.shape, y_train.shape)
print("Testing Set:", X_test.shape, y_test.shape)
print("Validation Set:", X_val.shape, y_val.shape)
