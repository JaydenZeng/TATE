import librosa
import numpy as np
import os
import pickle

filepath = "./WAV_16000/Segmented"
files = os.listdir(filepath)
print (len(files))

audio_dict = {}
a= 0
for file in files:
    tmp_name = './WAV_16000/Segmented/' + file    
    y, sr = librosa.load(tmp_name, sr=None)
    hops = 512

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    f0 = librosa.feature.zero_crossing_rate(y, hops)

    mfcc = librosa.feature.mfcc(y, sr)

    cqt = librosa.feature.chroma_cqt(y_harmonic, sr)

    features = np.transpose(np.concatenate([f0,mfcc,cqt], 0))
    audio_dict[file[:-4]] = features
    a+=1
    if a %10 == 0:
        print (a)
pickle.dump(audio_dict, open('./audio_dict.pkl', 'wb'))
print (np.shape(audio_dict))