import time
import queue
import Levenshtein as Lev 
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
import argparse
parser = argparse.ArgumentParser()
#parser = argparse.ArgumentParser(desc5iption='Speech hackathon Baseline')
parser.add_argument('--infor', type=str, default='?')
parser.add_argument('--log', type=str, default='False')
#parser.add_argument("--pause", type=int, default=0)
from tensorboardX import SummaryWriter
summary = SummaryWriter()
args = parser.parse_args()
import random
import spec_augment_pytorch
import sparse_image_warp_pytorch

def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=50, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train()

    #print('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            pass
            #print('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()
        
        new_feats = feats.numpy()
        new_feats = np.transpose(new_feats, (0, 2, 1))
        
        
        LB = spec_augment_pytorch.spec_augment(mel_spectrogram=new_feats, frequency_mask_num=1) #SpecAugment
        LD = spec_augment_pytorch.spec_augment(mel_spectrogram=new_feats, frequency_mask_num=2) #SpecAugment
        OG = new_feats #Original
        
        gathered = np.concatenate((LB,LD,OG), axis=0)
        feats = np.transpose(gathered, (0, 2, 1))
        feats = torch.from_numpy(feats)
        #print("final input features are ", feats.size())
        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            #print('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)

        src_len = scripts.size(1)
        #target = scripts[:, 1:]
        target = scripts[:, 1:].clone()

        logit = model(feats, scripts[:,:-1], mode='train')

        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        dist, length = get_distance(target, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += target.size(0)

        loss.backward()
        optimizer.step_and_update_lr()
        
        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0
            
            print('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            
            summary.add_scalar('train_loss', total_loss / total_num, train.cumulative_batch_count)
            summary.add_scalar('train_cer', total_dist / total_length, train.cumulative_batch_count)
            begin = time.time()

            
        batch += 1
        train.cumulative_batch_count += 1

    #print('train() completed')
    return total_loss / total_num, total_dist / total_length

train.cumulative_batch_count = 0
def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents
def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 


def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length


def evaluate(model, dataloader, queue, criterion, device):
    #logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            logit = model(feats, scripts[:,:-1], mode='eval')
            logit = logit[:,:target.size(1), :]

            y_hat = logit.max(-1)[1]

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    #logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

import label_loader
import random
from loader import *

import torch
import torch.nn as nn
import torch.optim as optim

import queue

from models.transformer import Model  # 2d mel style vgg
#from models.transformer_3d import Model # 3d CNN
from models.utils import ScheduledOptim, LabelSmoothingLoss

DATASET_PATH = '/mnt/junewoo/naver/all_dataset'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


char2index, index2char = label_loader.load_label("hackathon.labels")
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

char2index['[MASK]'] = len(char2index)
index2char[len(index2char)] = '[MASK]'

MASK_token = char2index['[MASK]']

##############################################################
#feature_size = N_FFT / 2 + 1#N_FFT: defined in loader.py
feature_size = N_FFT / 2 #N_FFT: defined in loader.py

batch_size = 12
epochs = 200

teacher_forcing = True
lr = 5e-8
##############################################################


def split_dataset( wav_paths, script_paths, valid_ratio=0.05):
    train_loader_count = 3
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / 3)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(3):

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin *batch_size
        train_end_raw_id = train_end * batch_size
        
        #print("train set load start...")
        #print("-----------------------")
        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id],
                                        script_paths[train_begin_raw_id:train_end_raw_id],
                                        SOS_token, EOS_token))
        train_begin = train_end 

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)
    #print("valid...")
    #print("-----------------------")

    return train_batch_num, train_dataset_list, valid_dataset


#d_model_size = 1536
d_model_size = 512
model = Model(len(char2index), SOS_token, EOS_token, d_model=d_model_size, nhead=8, max_seq_len=1024, 
                                                         num_encoder_layers=0, num_decoder_layers=4,
                                                         enc_feedforward=2048, dec_feedforward=2048,
                                                         dropout=0.1, padding_idx=PAD_token, mask_idx=MASK_token, device=device)

    
model.to(device)

optimizer = ScheduledOptim(optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-3),
            d_model_size, 4000)

#criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
#criterion = nn.NLLLoss(reduction='sum', ignore_index=PAD_token).to(device)
criterion = LabelSmoothingLoss(0.1, len(char2index), ignore_index=PAD_token).to(device)

#**************************************************************************
data_list = os.path.join(DATASET_PATH, 'train/train_data', 'data_list.csv')
wav_paths = list()
script_paths = list()

with open(data_list, 'r') as f:
    for line in f:
        # line: "aaa.wav,aaa.label"

        wav_path, script_path = line.strip().split(',')
        wav_paths.append(os.path.join(DATASET_PATH, 'train/train_data', wav_path))
        script_paths.append(os.path.join(DATASET_PATH, 'train/train_data', script_path))
target_path = os.path.join(DATASET_PATH, 'train/train_label')

load_targets(target_path)

#**************************************************************************
train_batch_num, train_dataset_list, valid_dataset = split_dataset(wav_paths, script_paths, valid_ratio=0.05)



train_begin = time.time()
for epoch in range(epochs):
    
    train_queue = queue.Queue(3 * 2)
    
    train_loader = MultiLoader(train_dataset_list, train_queue, batch_size, 3)
    train_loader.start()
    
    train_loss, train_cer = train(model, train_batch_num, train_queue, criterion, optimizer, device, train_begin, 3, 10, teacher_forcing)
    print('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
    
    train_loader.join()
    
    valid_queue = queue.Queue(3 * 2)
    valid_loader = BaseDataLoader(valid_dataset, valid_queue, batch_size, 0)
    valid_loader.start()

    eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
    print('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))
    summary.add_scalar('eval_loss', eval_loss, epoch)
    summary.add_scalar('eval_cer', eval_cer, epoch)
    

    valid_loader.join()
