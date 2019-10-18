import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_sinusoid_encoding_table

import copy
import math
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Gelu(nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()
        
    def forward(self, x):
        
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        

#def gelu(nn.Module):
#    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):   
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.attn_score = self.self_attn(src, src, src, attn_mask=src_mask,
                                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
       
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                      tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, self.attn_score = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, 
                                                        key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, self.enc_dec_attn_score = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                       num_decoder_layers=6, enc_feedforward=2048, dec_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, enc_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dec_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                      src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(            
            nn.Conv2d(1,64,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.block2 = nn.Sequential(            
            nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.block3 = nn.Sequential(            
            nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.block4 = nn.Sequential(            
            nn.Conv2d(64,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.block5 = nn.Sequential(            
            nn.Conv2d(128,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.block6 = nn.Sequential(            
            nn.Conv2d(128,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.block7 = nn.Sequential(            
            nn.Conv2d(128,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(kernel_size=(2,4), stride=(2,4))

    def forward(self, x):
        x = self.block1(x) + x[:,:,2:-2,2:-2]
        x = self.block2(x) + x[:,:,2:-2,2:-2]
        x = self.block3(x) + x[:,:,2:-2,2:-2]
        x = self.block4(x)
        x = self.block5(x) + x[:,:,2:-2,2:-2]
        x = self.block6(x) + x[:,:,2:-2,2:-2]
        x = self.block7(x) + x[:,:,2:-2,2:-2]
        x = self.pool(x)
        return x


class Model(nn.Module):
    def __init__(self, vocab_len, sos_id, eos_id, d_model=512, nhead=8, num_encoder_layers=6, 
                       num_decoder_layers=6, max_seq_len=512, enc_feedforward=2048, dec_feedforward=2048,
                       dropout=0.1, max_length=500, padding_idx=0, mask_idx=0, device=None):
        super(Model, self).__init__()
        self.device = device
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.max_seq_len = max_seq_len
        self.max_length = max_length
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # mel 80, like paper SpecAugment (not done)
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1)), #78
            #nn.Conv3d(1, 64, kernel_size=(), stridr=(), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)), #76
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)), # @18 23 22
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)), #16 20
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)), #14 18
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)), #12 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)), #6 8
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)), #4 6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)), #2 4 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)) #2
            #nn.Dropout(0.3)
        )
        
        
        #3d
        '''
        self.conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1)), #64, 64, 15, T, 38
            #nn.Conv3d(1, 64, kernel_size=(), stridr=(), 
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1)), #64, 64, 15, T, 36
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)), #64, 64, 15, T, 18
            nn.Dropout(0.3),

            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1)), #64, 128, 15, T, 16
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1)), #64, 128, 15, T, 14
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1)), #64, 128, 15, T, 12
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)), #64, 128, 15, T, 6
            nn.Dropout(0.3),

            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1)), #64, 256, 15, T, 4
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1)), #64, 256, 15, T, 2 
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 1), stride=(1, 2, 1)), #64, 256, 15, T, 2
            #nn.Dropout(0.3),
        )
        '''
        
        
               
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, enc_feedforward=enc_feedforward, 
                                       dec_feedforward=dec_feedforward, dropout=dropout)

        self.embedding = nn.Embedding(vocab_len, d_model, padding_idx=padding_idx)
        self.enc_pos_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_seq_len+1, d_model, 
                                                                    padding_idx=padding_idx), freeze=True)
        self.dec_pos_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_length+1, d_model, 
                                                                    padding_idx=padding_idx), freeze=True)

        self.classifier = nn.Linear(d_model, vocab_len)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        #self.RNN = nn.LSTM(512,512,2,bidirectional=True) # num_layer=2
        self.RNN = nn.LSTM(512,512,6,bidirectional=True) # num_layer=3


    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                      src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, mode=None):
        batch_size = src.size(0)
        divide_num = 16
        src = src[:,:self.max_seq_len]
        src_sizes = src.size()
        src_list = list(src_sizes)
        src = src.view(src_sizes[0], divide_num, src_list[1]//divide_num, src_sizes[2])  
        src = self.conv(src.unsqueeze(1))
        sizes = src.size()
        src = src.view(sizes[0], sizes[1], sizes[2] * sizes[3], sizes[4])
        src = src.transpose(1, 2)
        src = src.contiguous()
        sizes = src.size()
        src = src.view(sizes[0], sizes[1], sizes[2] * sizes[3]) #3D, (batch, seq_len, size)
        src = src.transpose(1, 0) #3D, (seq_len, batch, size)
        
        #src_pos = torch.LongTensor(range(src.size(1))).to(self.device)
        #src = src + self.enc_pos_enc(src_pos)
        src = self.RNN(src)
        src = src[0]
        src = src.transpose(0, 1) 
        
        if mode == 'train':
            # Enhance Decoder's representation
            mask_p = 0.2
            nonzero = torch.nonzero(tgt)
            rand = torch.randint(nonzero.size(0), (1,int(nonzero.size(0)*mask_p)))[0]
            nonzero = nonzero[rand].split(1, dim=1)
            tgt[nonzero] = self.mask_idx

            tgt_key_padding_mask = (tgt == self.padding_idx).to(self.device)
            tgt_pos = torch.LongTensor(range(tgt.size(1))).to(self.device)
            tgt = self.embedding(tgt) + self.dec_pos_enc(tgt_pos)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
            
            output = self.transformer(src.transpose(0,1), tgt.transpose(0,1), src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                        src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask).transpose(0,1)

            output = self.classifier(output)
            output = self.logsoftmax(output)

            return output

        else:
            memory = self.transformer.encoder(src.transpose(0,1), mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            if tgt is None: # Inferenece
                tgt = torch.LongTensor([[self.sos_id]]).to(self.device)
                for di in range(100):
                    tgt_pos = torch.LongTensor(range(tgt.size(1))).to(self.device)
                    tgt_ = self.embedding(tgt) + self.dec_pos_enc(tgt_pos)
                    tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                    output = self.transformer.decoder(tgt_.transpose(0,1), memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask).transpose(0,1)
                    
                    output = self.classifier(output)
                    output = self.logsoftmax(output)
                    symbols = torch.max(output, -1)[1][:,-1].unsqueeze(1)
                    tgt = torch.cat((tgt,symbols),1)
            else: # Evaluate
                tgt_size = tgt.size(1)
                tgt = tgt[:,0].unsqueeze(1)
                for di in range(tgt_size):
                    tgt_pos = torch.LongTensor(range(tgt.size(1))).to(self.device)
                    tgt_ = self.embedding(tgt) + self.dec_pos_enc(tgt_pos)
                    tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                    output = self.transformer.decoder(tgt_.transpose(0,1), memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask).transpose(0,1)
                    
                    output = self.classifier(output)
                    output = self.logsoftmax(output)
                    symbols = torch.max(output, -1)[1][:,-1].unsqueeze(1)
                    tgt = torch.cat((tgt,symbols),1)
                
            return output
