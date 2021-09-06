import logging
import os
import sys
import glob
from tqdm import tqdm
import numpy as np
#import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class FtsNet2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 5
        padding = (kernel_size-1)//2
        
        self.conv1 = torch.nn.Conv2d(1,   128, kernel_size, padding=padding, padding_mode='circular')
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv8 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv9 = torch.nn.Conv2d(128, 128, kernel_size, padding=padding, padding_mode='circular')
        self.conv10 = torch.nn.Conv2d(128,   1, 1)

        # self.bn1 = torch.nn.BatchNorm3d(128)
        # self.bn2 = torch.nn.BatchNorm3d(128)
        # self.bn3 = torch.nn.BatchNorm3d(128)
        # self.bn4 = torch.nn.BatchNorm3d(128)
        # self.bn5 = torch.nn.BatchNorm3d(128)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = torch.nn.functional.relu(self.conv7(x))
        x = torch.nn.functional.relu(self.conv8(x))
        x = torch.nn.functional.relu(self.conv9(x))
        x = self.conv10(x)
        return x

class DFTS_Dataset(Dataset):
    def __init__(self, data_dir):

        file_list = glob.glob(data_dir + "/*.npz")
        sample_data = np.load(file_list[0])
        nx = sample_data["nx"]
        self.nx = nx
        
        n_train = len(file_list)
        self.__n_train = n_train
        self.__X = np.zeros([n_train, 1, nx[0], nx[1]])
        self.__Y = np.zeros([n_train, 1, nx[0], nx[1]])

        # train data
        for i in range(0, n_train):
            data = np.load(file_list[i])
            # exchange field
            self.__X[i,0,:,:] = np.reshape(data["w_minus"],nx)/data["N"]
            # pressure field
            self.__Y[i,0,:,:] = np.reshape(data["w_plus"],nx)/data["N"]
            
        logging.info(f'{data_dir} X.shape{self.__X.shape}')
        logging.info(f'{data_dir} Y.shape{self.__Y.shape}')
        
        #logging.info(f'Creating dataset with {len(self.ids)} examples')
        
    def __len__(self):
        return self.__n_train
    
    def __getitem__(self, i):
        return {
            'data':   torch.tensor(self.__X[i], dtype=torch.float64),
            'target': torch.tensor(self.__Y[i], dtype=torch.float64)
        }

class DeepFts2d:
    def __init__(self, load_net=None):

        # os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        logging.info(f'Current cuda device {torch.cuda.current_device()}')
        logging.info(f'Count of using GPUs {torch.cuda.device_count()}')
        
        self.train_folder_name = "data2D/train"
        self.test_folder_name = "data2D/eval"
        
        self.net = FtsNet2d()
        self.net.to(device=self.device)
        
        if load_net:
            self.net.load_state_dict(torch.load(load_net, map_location=self.device))
            logging.info(f'Model loaded from {load_net}')
        
        self.net.double()
            
    def generate_w_plus(self, w_minus, nx):
        data = np.reshape(w_minus, (1, 1, nx[0], nx[1]))
        data = torch.tensor(data, dtype=torch.float64).to(self.device)
        #print(type(data), data.shape)
        with torch.no_grad():
            return np.reshape(self.net(data).cpu().numpy(), nx[0]*nx[1])
            
    def eval_net(self, net, loader, device, criterion, writer, global_step):
        net.eval()
        n_val = len(loader)
        loss = 0

        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in loader:        
                data = batch["data"].to(device)
                target = batch["target"].to(device)  
                with torch.no_grad():
                    y_pred = net(data)
                loss += criterion(y_pred, target).item()
                pbar.update()

            writer.add_images('exchange', data[0,0,:,:], global_step, dataformats="HW")
            writer.add_images('pred', y_pred[0,0,:,:], global_step, dataformats="HW")
            writer.add_images('true', target[0,0,:,:], global_step, dataformats="HW")

        net.train()
        return loss / n_val

    def train_net(self,) :
        
        net = self.net
        device = self.device
        
        lr = 1e-4
        epochs = 50
        batch_size = 20
        log_dir = "logs"
        output_dir = "checkpoints"
                
        train = DFTS_Dataset(self.train_folder_name)
        val = DFTS_Dataset(self.test_folder_name)
        
        n_train = len(train)
        n_val = len(val)
        
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, drop_last=False)    
        writer = SummaryWriter(log_dir=log_dir, comment=f'LR_{lr}_BS_{batch_size}')

        global_step = 0
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {output_dir}
            Device:          {device.type}
            Optimizer        {optimizer.__class__.__name__}
            Criterion        {criterion.__class__.__name__}
        ''')
        
        global_step = 0
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='data') as pbar:
                for batch in train_loader:                
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)              
                    y_pred = net(data)
                    loss = criterion(y_pred, target)
                    epoch_loss += loss.item()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
            
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.update(data.shape[0])
                    
                    global_step += 1
                    if global_step % (n_train // (batch_size)) == 0:
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        val_loss = self.eval_net(net, val_loader, device, criterion, writer, global_step)
                        logging.info('Validation loss: {}'.format(val_loss))
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                        writer.add_scalar('Loss/test', val_loss, global_step)
            try:
                os.mkdir(output_dir)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(output_dir, f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')
        torch.cuda.empty_cache()

if __name__ == '__main__':
    model = DeepFts2d()
    #model = DeepFts2d("checkpoints/CP_epoch50.pth")
    model.train_net()
    
    sample_file_name = "data2D/eval/fields_050000.npz"
    sample_data = np.load(sample_file_name)
    nx = sample_data["nx"]
    X = np.reshape(sample_data["w_minus"], (1, 1, nx[0], nx[1]))/sample_data["N"]
    Y = np.reshape(sample_data["w_plus"],  (1, 1, nx[0], nx[1]))/sample_data["N"]
    Y_gen = np.reshape(model.generate_w_plus(X, (nx[0], nx[1])), (1, 1, nx[0], nx[1]))
    vmin = np.min([np.min(Y), np.min(Y_gen)])
    vmax = np.max([np.max(Y), np.max(Y_gen)])
    
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    axes[0,0].axis("off")
    axes[0,1].axis("off")
    axes[1,0].axis("off")
    axes[1,1].axis("off")
    
    axes[0,0].imshow(X[0,0,:,:], cmap="jet")
    axes[1,0].imshow(Y    [0,0,:,:], vmin=vmin, vmax=vmax, cmap="jet")
    axes[1,1].imshow(Y_gen[0,0,:,:], vmin=vmin, vmax=vmax, cmap="jet")
    
    plt.subplots_adjust(left=0.01,bottom=0.01,
                        top=0.99,right=0.99,
                        wspace=0.01, hspace=0.01)
    plt.savefig('w_plus_minus.png')
