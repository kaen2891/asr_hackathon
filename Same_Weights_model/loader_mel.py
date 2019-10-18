"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
#import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np
from warnings import warn

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000
#hoplen = int(N_FFT/2)
target_dict = dict()
n_mels = 40

win_length=int(0.032*SAMPLE_RATE) #change 0930_1628
hop_length=int(0.016*SAMPLE_RATE) #change 0930_1628

#win_length=int(0.030*SAMPLE_RATE) #basic
#hop_length=int(0.01*SAMPLE_RATE) #basic


def manipulate_shift(data, sampling_rate, shift_max=0.5, shift_direction= "both"):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift    
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def manipulate_pitch(data, sampling_rate, pitch_factor=np.random.uniform(low=-4, high=4, size=(1,)).item()):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def manipulate_time(data, speed_factor=np.random.uniform(low=0.85, high=1.2, size=(1,)).item()):
    return librosa.effects.time_stretch(data, speed_factor)

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target
'''wav norm'''
'''
def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()
'''    
def get_mel_feature(filepath):
    y, fs = librosa.load(filepath, sr=SAMPLE_RATE)
    S = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=n_mels, n_fft=win_length, hop_length=hop_length)
    feat = torch.FloatTensor(S).transpose(0, 1)
    #feat = normalize(feat)
    
    if feat.size(0) > 1024:
        feat = feat[:1024, :]
    if feat.size(0) < 288:    
        zero = torch.zeros([288, feat.size(1)])
        feat = torch.cat([feat, zero[:zero.size(0) - feat.size(0), :]], axis=0)
        #feat = feat[:288, :]
    
    divide_num = 16
    seqs = 0
    if feat.size(0) % divide_num !=0:
        seqs = ((feat.size(0)//divide_num))*divide_num
    if seqs != 0:
        feat = feat[:seqs, :]
        #val.fill_(PAD)
    
    return feat
    
def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308, train=False):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.train = train
    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        
        feat = get_mel_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        return feat, script

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    divide_num = 16
    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    #print("max_seq_sample is ", np.shape(max_seq_sample))
    max_target_sample = max(batch, key=target_length_)[1]
    #print("max_target_sample is", np.shape(max_target_sample))

    max_seq_size = max_seq_sample.size(0)
    #print("max_seq_size is", max_seq_size)
    
    ### add
    if max_seq_size % divide_num !=0:
        max_seq_size = ((max_seq_size//divide_num)+1)*divide_num
    
    max_target_size = len(max_target_sample)
    #print("max_target_size is", max_target_size)

    feat_size = max_seq_sample.size(1)
    #print("feat_size is", feat_size)
    batch_size = len(batch)
    #print("batch_size is", batch_size)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    #print("seqs is", seqs.size())
    #print("end of loader.....")
    #print("------------")

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()
