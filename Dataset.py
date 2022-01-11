# -*- coding: UTF-8 -*-
from Settings import Config
import re
import os
import sys
import numpy as np
import pickle
import random

class Dataset:
    def __init__(self):
        self.config = Config()
        self.iter_num = 0
        self.label_set = ['Negative', 'Neutral', 'Positive']
        self.visual = pickle.load(open('../data/mosi/processed_visual_dict.pkl', 'rb'))
        self.audio = pickle.load(open('../data/mosi/audio_dict.pkl', 'rb'))
        self.text = pickle.load(open('../data/mosi/text_emb.pkl', 'rb'))
        self.label = pickle.load(open('../data/mosi/label_dict.pkl', 'rb'))
        self.max_visual_len = self.config.max_visual_len
        self.max_audio_len = self.config.max_audio_len
        self.max_text_len = self.config.max_text_len
        self.keys = []
        for key in self.label:
            self.keys.append(key)
        np.random.shuffle(self.keys)
        self.flag = {}
        for item in self.keys:
            self.flag[item] = -1



    def padding(self, seq, max_len):
        shape_1 = np.shape(seq)[0]
        shape_2 = np.shape(seq)[1]
        emb_matrix = list(np.zeros([max_len, shape_2]))
        if shape_1 >= max_len:
            shape_1 = max_len
        for i in range(shape_1):
            emb_matrix[i] = seq[i]
        return emb_matrix

    def miss_modality(self, lost_rate):
        miss_visual = list(np.zeros([self.max_visual_len, 709]))
        miss_audio = list(np.zeros([self.max_audio_len, 33]))
        miss_text = list(np.zeros([self.max_text_len, 768]))
        #set miss value
        for i in range(0, len(self.keys), 100):
            for j in range(i, i + int(100*lost_rate)):
                rnd = random.randint(0, 3)  # 0: visual  1:audio 2:text
                if rnd == 0:
                   self.visual[self.keys[j]] = miss_visual
                   self.flag[self.keys[j]] = 0
                elif rnd == 1:
                   self.audio[self.keys[j]] = miss_audio
                   self.flag[self.keys[j]] = 1
                elif rnd == 2:
                   self.text[self.keys[j]] = miss_text
                   self.flag[self.keys[j]] = 2
                elif rnd == 3:
                   self.audio[self.keys[j]] = miss_audio
                   self.text[self.keys[j]] = miss_text
                   self.flag[self.keys[j]] = 3
                elif rnd == 4:
                   self.visual[self.keys[j]] = miss_visual
                   self.text[self.keys[j]] = miss_text
                   self.flag[self.keys[j]] = 4
                else:
                   self.visual[self.keys[j]] = miss_visual
                   self.audio[self.keys[j]] = miss_audio
                   self.flag[self.keys[j]] = 5



    def setdata(self, train_size):
        traindata = {'ID':[], 'V':[], 'A':[], 'T':[], 'L':[], 'F':[]}
        testdata = {'ID':[], 'V':[], 'A':[], 'T':[], 'L':[], 'F':[]}
        temp_order = list(range(len(self.label)))
#        np.random.shuffle(temp_order)
        for i in range(len(self.keys)):
            cur_id = self.keys[i]
            if i < train_size:
                traindata['ID'].append(cur_id)
                traindata['V'].append(self.padding(self.visual[cur_id], self.max_visual_len))
                traindata['A'].append(self.padding(self.audio[cur_id], self.max_audio_len))
                traindata['T'].append(self.padding(self.text[cur_id], self.max_text_len))
                traindata['L'].append(self.label_set.index(self.label[cur_id]))
                traindata['F'].append(self.flag[cur_id])
            else:
                testdata['ID'].append(cur_id)
                testdata['V'].append(self.padding(self.visual[cur_id], self.max_visual_len))
                testdata['A'].append(self.padding(self.audio[cur_id], self.max_audio_len))
                testdata['T'].append(self.padding(self.text[cur_id], self.max_text_len))
                testdata['L'].append(self.label_set.index(self.label[cur_id]))
                testdata['F'].append(self.flag[cur_id])
        return traindata, testdata

    def nextBatch(self, traindata, testdata, is_training = True):
        nextIDBatch = []
        nextVisualBatch = []
        nextAudioBatch = []
        nextTextBatch = []
        nextLabelBatch = []
        nextFlagBatch = []

        if is_training:
            if (self.iter_num+1)*self.config.batch_size > len(traindata['ID']):
                self.iter_num = 0
            if self.iter_num == 0:
                self.temp_order =  list(range(len(traindata['ID'])))
                np.random.shuffle(self.temp_order)
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]
        else:
            if (self.iter_num+1)*self.config.batch_size > len(testdata['ID']):
                self.iter_num = 0
            if self.iter_num == 0:
                self.temp_order =  list(range(len(testdata['ID'])))
            temp_order = self.temp_order[self.iter_num*self.config.batch_size:(self.iter_num+1)*self.config.batch_size]


        ID = []
        visual = []
        audio = []
        text = []
        label = []
        flag = []

        for it in temp_order:
            if is_training:
                ID.append(traindata['ID'][it])
                visual.append(traindata['V'][it])
                audio.append(traindata['A'][it])
                text.append(traindata['T'][it])
                label.append(traindata['L'][it])
                flag.append(traindata['F'][it])
            else:
                ID.append(testdata['ID'][it])
                visual.append(testdata['V'][it])
                audio.append(testdata['A'][it])
                text.append(testdata['T'][it])
                label.append(testdata['L'][it])
                flag.append(testdata['F'][it])

        self.iter_num += 1
        nextIDBatch = np.array(ID)
        nextVisualBatch = np.array(visual)
        nextAudioBatch = np.array(audio)
        nextTextBatch = np.array(text)
        nextLableBatch = np.array(label)
        nextFlagBatch = np.array(flag)

        cur_batch = {'ID':nextIDBatch, 'V':nextVisualBatch, 'A':nextAudioBatch, 'T':nextTextBatch, 'L':nextLableBatch, 'F':nextFlagBatch}
        return cur_batch

if __name__ == '__main__':
    data = Dataset()
    data.miss_modality(data.config.miss_rate)
    print (data.config.miss_rate)   
    traindata, testdata = data.setdata(2000)

#    print (traindata['A'][:2])

    pickle.dump(traindata, open('./train.pkl', 'wb'))
    pickle.dump(testdata, open('./test.pkl', 'wb'))


#    print (traindata['L'][:10])
    cur_batch = data.nextBatch(traindata, testdata, True)
#    print (cur_batch['L'])
    print (cur_batch['ID'])
