# ====================================================
# Library
# ====================================================
import os
import gc
import sys
import json
import time
import math
import random
from datetime import datetime
from collections import Counter, defaultdict

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from tqdm.auto import tqdm
import category_encoders as ce

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")

# ====================================================
# CFG
# ====================================================
class CFG:
    competition='ventilator'
    _wandb_kernel='nakama'
    apex=False
    print_freq=100
    num_workers=4
    model_name='rnn'
    scheduler='CosineAnnealingLR' # ['linear', 'cosine', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    batch_scheduler=False
    #num_warmup_steps=100 # ['linear', 'cosine']
    #num_cycles=0.5 # 'cosine'
    #factor=0.2 # ReduceLROnPlateau
    #patience=4 # ReduceLROnPlateau
    #eps=1e-6 # ReduceLROnPlateau
    T_max=50 # CosineAnnealingLR
    #T_0=50 # CosineAnnealingWarmRestarts
    epochs=5
    max_grad_norm=1000
    gradient_accumulation_steps=1
    hidden_size=64
    lr=5e-3
    min_lr=1e-6
    weight_decay=1e-6
    batch_size=64
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    cate_seq_cols=['R', 'C']
    cont_seq_cols=['time_step', 'u_in', 'u_out'] + ['breath_time', 'u_in_time']
    train=True
    inference=True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby('breath_id').groups
        self.keys = list(self.groups.keys())
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        cate_seq_x = torch.LongTensor(df[CFG.cate_seq_cols].values)
        cont_seq_x = torch.FloatTensor(df[CFG.cont_seq_cols].values)
        u_out = torch.LongTensor(df['u_out'].values)
        label = torch.FloatTensor(df['pressure'].values)
        return cate_seq_x, cont_seq_x, u_out, label
    

class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby('breath_id').groups
        self.keys = list(self.groups.keys())
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indexes = self.groups[self.keys[idx]]
        df = self.df.iloc[indexes]
        cate_seq_x = torch.LongTensor(df[CFG.cate_seq_cols].values)
        cont_seq_x = torch.FloatTensor(df[CFG.cont_seq_cols].values)
        return cate_seq_x, cont_seq_x