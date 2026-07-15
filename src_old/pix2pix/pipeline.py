#==============================================================================
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
#==============================================================================
def forward_transform_aia(x, U=14, L=0):
    mid = (U + L) / 2
    thr = (U - L) / 2
    y = np.log2(x + 1)
    z = (y - mid) / thr
    return z

def inverse_transform_aia(z, U=14, L=0):
    mid = (U + L) / 2
    thr = (U - L) / 2
    y = z * thr + mid
    x = 2**y - 1
    return x

def forward_transform_hmi(x, s=1500):
    z = x / s
    return z

def inverse_transform_hmi(z, s=1500):
    x = z * s
    return x

#==============================================================================
class PairedDataset(Dataset):
    def __init__(self, data_root, data_csv, input_cols, target_cols):
        super().__init__()
        df = pd.read_csv(data_csv)
        filenames = df['filename'].tolist()
        self.files = [Path(data_root) / filename for filename in filenames]
        self.input_cols = input_cols.split(',')
        self.target_cols = target_cols.split(',')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        input_list = []
        for input_col in self.input_cols:
            inp_data = data[input_col]
            if 'aia' in input_col:
                inp_data = forward_transform_aia(inp_data)
            elif 'hmi' in input_col:
                inp_data = forward_transform_hmi(inp_data)
            input_list.append(inp_data)
        inputs = np.stack(input_list, axis=0)
        inputs = torch.from_numpy(inputs.astype(np.float32)).float()
        
        target_list = []
        for target_col in self.target_cols:
            trg_data = data[target_col]
            if 'aia' in target_col:
                trg_data = forward_transform_aia(trg_data)
            elif 'hmi' in target_col:
                trg_data = forward_transform_hmi(trg_data)
            target_list.append(trg_data)
        target = np.stack(target_list, axis=0)
        target = torch.from_numpy(target.astype(np.float32)).float()
        
        return inputs, target
    
#==============================================================================