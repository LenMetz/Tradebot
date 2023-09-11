import torch
import numpy as np

### data loader and dataset ###
class Prices(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_length, shift = 100, one_sample = False, center=True, norm=False, tgt_step = 5):
        self.seq_length = seq_length
        self.data = np.load(data_path)
        self.data_size = len(self.data)
        self.shift = shift # shift of each data sample so samples aren't too close together
        self.one_sample = one_sample # return only one sample
        self.center = center # zero data around last point in src
        self.norm = norm # zero data around last point in src
        self.tgt_step=tgt_step # size of target step
        
    def __len__(self):
        return int((self.data_size-self.seq_length-self.tgt_step)/self.shift)
    
    def __getitem__(self, index):
        if self.one_sample:
            src = torch.tensor(self.data[0:self.seq_length]).float()
            tgt = torch.tensor(self.data[self.seq_length:self.seq_length+self.tgt_len]).float()
        else:
            src = torch.tensor(self.data[self.shift*index:self.shift*index+self.seq_length]).float()
            tgt = torch.tensor(self.data[self.shift*index+self.seq_length+self.tgt_step-1]).float()
        if self.center:
            src_last = src[-1]
            src = src-src_last
            tgt = tgt-src_last
        if self.norm:
            src_max = torch.max(torch.abs(src))
            src = (1/src_max)*src
            tgt = (1/src_max)*tgt
        return src, tgt

class PricesVol(torch.utils.data.Dataset):
    def __init__(self, data_path, data_path_vol, seq_length, shift = 100, one_sample = False, center=True, norm=False, tgt_step = 5):
        self.seq_length = seq_length
        self.data = np.load(data_path)
        self.data_vol = np.load(data_path_vol)
        self.data_size = len(self.data)
        self.shift = shift # shift of each data sample so samples aren't too close together
        self.one_sample = one_sample # return only one sample
        self.center = center # zero data around last point in src
        self.norm = norm # zero data around last point in src
        self.tgt_step=tgt_step # size of target step
        
    def __len__(self):
        return int((self.data_size-self.seq_length-self.tgt_step)/self.shift)
    
    def __getitem__(self, index):
        if self.one_sample:
            src = torch.tensor(self.data[0:self.seq_length]).float()
            vol = torch.tensor(self.data_vol[0:self.seq_length]).float()
            tgt = torch.tensor(self.data[self.seq_length:self.seq_length+self.tgt_len]).float()
        else:
            src = torch.tensor(self.data[self.shift*index:self.shift*index+self.seq_length]).float()
            vol = torch.tensor(self.data_vol[self.shift*index:self.shift*index+self.seq_length]).float()
            tgt = torch.tensor(self.data[self.shift*index+self.seq_length+self.tgt_step-1]).float()
        if self.center:
            src_last = src[-1]
            src = src-src_last
            tgt = tgt-src_last
            vol = vol/max(1,torch.max(vol))
        if self.norm:
            src_max = torch.max(torch.abs(src))
            src = (1/src_max)*src
            tgt = (1/src_max)*tgt
        return src, vol, tgt