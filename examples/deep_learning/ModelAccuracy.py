import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from fts_dataset2d import *

#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
train_folder_name = "data2D/train"
test_folder_name = "data2D/eval"
#model_file = "checkpoints/CP_epoch100.pth"
model_file = "temp_models/FCN2d_5Layer_5kernel_128channel_epoch100_mish.pth"
batch_size = 20

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
logging.info(f'Current cuda device {torch.cuda.current_device()}')
logging.info(f'Count of using GPUs {torch.cuda.device_count()}')

#net = FtsNet2d()
#net.to(device=device)
#net.load_state_dict(torch.load(model_file, map_location=device))
net = torch.load(model_file, map_location=device)
net.to(device=device)
logging.info(f'Model loaded from {model_file}')
net.double()

train = FtsDataset2d(train_folder_name)
val = FtsDataset2d(test_folder_name)

n_train = len(train)
n_val = len(val)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, drop_last=False)

net.eval()

criterion = torch.nn.MSELoss()
loss = 0
with tqdm.tqdm(total=n_train, desc='Training Data', unit='batch', leave=False) as pbar:
    for batch in train_loader:        
        data = batch["data"].to(device)
        target = batch["target"].to(device)  
        with torch.no_grad():
            y_pred = net(data)
        loss += criterion(y_pred, target).item()
        pbar.update(data.shape[0])

print(loss/n_train)

loss = 0
with tqdm.tqdm(total=n_val, desc='Test Data', unit='batch', leave=False) as pbar:
    for batch in val_loader:        
        data = batch["data"].to(device)
        target = batch["target"].to(device)  
        with torch.no_grad():
            y_pred = net(data)
        loss += criterion(y_pred, target).item()
        pbar.update(data.shape[0])

print(loss/n_val)

pytorch_total_params = sum(p.numel() for p in net.parameters())
pytorch_total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)

print(pytorch_total_params)
print(pytorch_total_params_train)
