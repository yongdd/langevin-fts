import logging
import os
import sys
import glob
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class DFTS_FCNet3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1,   128, 3, padding=1, padding_mode='circular')
        self.conv2 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv3 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv4 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv5 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv6 = torch.nn.Conv3d(128,   1, 1)

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
        x = self.conv6(x)
        return x

class DFTS_Dataset(Dataset):
    def __init__(self, data_dir):

        file_list = glob.glob(data_dir + "/*.npz")
        sample_data = np.load(file_list[0])
        nx = sample_data["nx"]
        
        n_train = len(file_list)
        self.__n_train = n_train
        self.__X_train = np.zeros([n_train, 1, nx[0], nx[1], nx[2]])
        self.__Y_train = np.zeros([n_train, 1, nx[0], nx[1], nx[2]])

        # train data
        for i in range(0, n_train):
            data = np.load(file_list[i])
            # exchange field
            self.__X_train[i,0,:,:,:] = np.reshape(data["w_minus"],nx)/data["N"]
            # pressure field
            self.__Y_train[i,0,:,:,:] = np.reshape(data["w_plus"],nx)/data["N"]
            
        print(self.__X_train.shape)
        print(self.__Y_train.shape)
        
        #logging.info(f'Creating dataset with {len(self.ids)} examples')
        
    def __len__(self):
        return self.__n_train
    
    def __getitem__(self, i):
        
        self.__flip(self.__X_train[i], self.__Y_train[i])
        
        return {
            'data':   torch.tensor(self.__X_train[i], dtype=torch.float32),
            'target': torch.tensor(self.__Y_train[i], dtype=torch.float32)
        }
    
    def __flip(self, data, target):
        rand_number = np.random.randint(4)
        #if(rand_number == 0):
        # do nothing
        if(rand_number == 1):
            data[:,:,:,:] = data[:,::-1,:,:]
            target[:,:,:,:] = target[:,::-1,:,:]
        elif(rand_number == 2):
            data[:,:,:,:] = data[:,:,::-1,:]
            target[:,:,:,:] = target[:,:,::-1,:]
        elif(rand_number == 3):
            data[:,:,:,:] = data[:,:,:,::-1]
            target[:,:,:,:] = target[:,:,:,::-1]

def eval_net(net, loader, device, criterion, writer, global_step):
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

        #print(data[:,0,0,:,:].shape)
        #print(y_pred[:,0,0,:,:].shape)
        #print(target[:,0,0,:,:].shape)
        
        writer.add_images('exchange', data[0,0,0,:,:], global_step, dataformats="HW")
        writer.add_images('pred', y_pred[0,0,0,:,:], global_step, dataformats="HW")
        writer.add_images('true', target[0,0,0,:,:], global_step, dataformats="HW")

    net.train()
    return loss / n_val

def train_net(net, device) :
    
    lr = 1e-4
    epochs = 20
    batch_size = 6
    output_dir = "checkpoints"
    
    train_folder_name = "data/train"
    test_folder_name = "data/eval"
    
    train = DFTS_Dataset(train_folder_name)
    val = DFTS_Dataset(test_folder_name)
    
    n_train = len(train)
    n_val = len(val)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)    
    writer = SummaryWriter(log_dir='./logs', comment=f'LR_{lr}_BS_{batch_size}')

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
                    val_loss = eval_net(net, val_loader, device, criterion, writer, global_step)
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
    
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(f'Current cuda device {torch.cuda.current_device()}')
    logging.info(f'Count of using GPUs {torch.cuda.device_count()}')
    
    net = DFTS_FCNet3D()
    net.to(device=device)
    
    # if cfg.load:
        # net.load_state_dict(
            # torch.load(cfg.load, map_location=device)
        # )
        # logging.info(f'Model loaded from {cfg.load}')
        
    try:
        train_net(net=net,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
