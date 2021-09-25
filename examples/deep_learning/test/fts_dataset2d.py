import numpy as np
import glob
import logging
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

class FtsDataset2d(Dataset):
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
            self.__X[i,0,:,:] = np.reshape(data["w_minus"],nx)/10.0
            # pressure field
            self.__Y[i,0,:,:] = np.reshape(data["w_plus"],nx)/10.0
            
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
