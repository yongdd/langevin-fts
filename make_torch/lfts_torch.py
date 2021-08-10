import torch
from torch.utils.data import  TensorDataset, DataLoader

import time
import matplotlib.pyplot as plt
import numpy as np
import glob, os
from pathlib import Path

class FCNet3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(2,   128, 3, padding=1, padding_mode='circular')
        self.conv2 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv3 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv4 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv5 = torch.nn.Conv3d(128, 128, 3, padding=1, padding_mode='circular')
        self.conv6 = torch.nn.Conv3d(128,   1, 1)

        #torch.nn.init.xavier_normal_(self.conv1.weight)
        #torch.nn.init.xavier_normal_(self.conv2.weight)
        #torch.nn.init.xavier_normal_(self.conv3.weight)
        #torch.nn.init.xavier_normal_(self.conv4.weight)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x)) 
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.conv6(x)
        return x

class DeepLFTS:
    def __init__(self, data_shape, boundary_condition, train_model, model_file = None):
        
        self.scale = 30
        self.lr = 1e-4
        self.batch_size = 10
        self.num_epoch = 5
        self.log_interval = 10
        (self.mx, self.my, self.mz)  = data_shape

        self.bc = boundary_condition 
        print(self.mx, self.my, self.mz)
        print(self.bc)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device('cuda:1')

        if (model_file == None):
            self.net = FCNet3D()
        else :
            self.net = torch.load(model_file, map_location=self.device)

        if (train_model == True):
            self.model_is_trained = False
        else :
            self.model_is_trained = True

        self.net.to(self.device)

        #model_comp = torch.load('backup/model_020000.pt', map_location=self.device)
        #for key in self.net.state_dict().keys():
        #    print(key, self.net.state_dict()[key].shape,
        #       torch.std(self.net.state_dict()[key] - model_comp.state_dict()[key]))
        #for param in self.net.parameters():
        #    print(param)

        print(self.net)

    def load_model_weights(self, model_file):

        self.net.load_state_dict(torch.load(model_file, map_location=self.device).state_dict(),strict=True)
        self.model_is_trained = True

    def train_model(self, folder_name, out_name) :

        mx = self.mx
        my = self.my
        mz = self.mz

        file_list = glob.glob(folder_name + "/*.npy")
        #print(type(file_list))
        #file_list.sort()
        #print(file_list)
        
        #n_train = int(len(file_list)*0.8)
        #n_test = len(file_list) - n_train
        
        n_train = len(file_list)-1
        n_test  = 1

        #n_train = 10
        #n_test = 10

        X_train = np.zeros([n_train, 2, mx, my, mz])
        X_test  = np.zeros([n_test,  2, mx, my, mz])
        Y_train = np.zeros([n_train, 1, mx, my, mz])
        Y_test  = np.zeros([n_test,  1, mx, my, mz])
        
        # train data
        for i in range(0,n_train):
            #print(i, file_list[i])
            fields = np.load(file_list[i])
            # exchange field
            X_train[i,0,:,:,:] = fields[0,:,:,:] /self.scale  
            # pressure field
            Y_train[i,0,:,:,:] = fields[1,:,:,:] /self.scale 

        # test data
        for i in range(0,n_test):
            fields = np.load(file_list[i + n_train])
            # exchange field
            X_test[i,0,:,:,:] = fields[0,:,:,:] /self.scale 
            # pressure field
            Y_test[i,0,:,:,:] = fields[1,:,:,:] /self.scale 
        
        X_train[:,1,:,:,:] = 1.0  
        X_test [:,1,:,:,:] = 1.0  
        if ( self.bc[0] == 0 ):
            print("self.bc[0] == 0")
            X_train[:,1, 0,:,:] =  0.0
            X_train[:,1,-1,:,:] =  0.0
            X_test [:,1, 0,:,:] =  0.0
            X_test [:,1,-1,:,:] =  0.0
        if ( self.bc[1] == 0 ):
            print("self.bc[1] == 0")
            X_train[:,1,:, 0,:] =  0.0
            X_train[:,1,:,-1,:] =  0.0
            X_test [:,1,:, 0,:] =  0.0
            X_test [:,1,:,-1,:] =  0.0
        if ( self.bc[2] == 0 ):
            print("self.bc[2] == 0")
            X_train[:,1,:,:, 0] =  0.0
            X_train[:,1,:,:,-1] =  0.0
            X_test [:,1,:,:, 0] =  0.0
            X_test [:,1,:,:,-1] =  0.0

        print(X_train.shape)
        print(Y_train.shape)
        
        print(X_train.shape)
        print(X_test.shape)

        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(Y_train, dtype=torch.float32)
        ds = TensorDataset(X, y)
        train_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        start_time = time.time()
        self.net.train()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        losses = []
        for epoch in range(self.num_epoch):
            batch_loss = 0.0
            batch_count = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                y_pred = self.net(data)
                loss = loss_fn(input=y_pred, target=target)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                batch_count += len(data)
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_count, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            losses.append(batch_loss)

        print(losses)

        torch.cuda.empty_cache()
        torch.save(self.net, out_name + '.pt')
        self.model_is_trained = True 
        self.net.eval()

        end_time = time.time()
        total_time = end_time - start_time
        print ("training time: ", total_time)

        print("test_file_name: ", file_list[n_train])
        with torch.no_grad():
            Y_pred = self.net(torch.Tensor(X_test[0:1,:,:,:,:]).to(self.device)).cpu().detach().numpy()
        '''
        plt.figure()
        plt.close()
        plt.plot(losses)
        plt.savefig(out_name + '_graph.png')
        plt.show()

        plt.figure()
        plt.imshow(X_test[0,0,:,:,25])
        plt.colorbar()
        plt.savefig(out_name + '_wminus.png')

        plt.figure()
        plt.imshow(X_test[0,1,:,:,25])
        plt.colorbar()
        plt.savefig(out_name + '_mask.png')

        plt.figure()
        plt.imshow(Y_pred[0,0,:,:,25])
        plt.colorbar()
        plt.savefig(out_name + '_out.png')

        plt.figure()
        plt.imshow(Y_test[0,0,:,:,25])
        plt.colorbar()
        plt.savefig(out_name + '_true.png')
        
        plt.figure()
        plt.imshow(Y_pred[0,0,:,:,25] - Y_test[0,0,:,:,25])
        plt.colorbar()
        plt.savefig(out_name + '_diff.png')
        plt.close()
        '''

    def predict(self, wminus) :
        self.net.eval()

        X_test  = np.zeros([1, 2, self.mx, self.my, self.mz])
        X_test[0,0,:,:,:] = wminus[:,:,:] /self.scale 
        X_test[0,1,:,:,:] = 1.0

        if ( self.bc[0] == 0 ):
            X_test [0,1, 0,:,:] =  0.0
            X_test [0,1,-1,:,:] =  0.0
        if ( self.bc[1] == 0 ):
            X_test [0,1,:, 0,:] =  0.0
            X_test [0,1,:,-1,:] =  0.0
        if ( self.bc[2] == 0 ):
            X_test [0,1,:,:, 0] =  0.0
            X_test [0,1,:,:,-1] =  0.0

        with torch.no_grad():
            wplus = self.net(torch.Tensor(X_test).to(self.device)).cpu().detach().numpy()*self.scale
            wplus = wplus.reshape(self.mx, self.my, self.mz)
            return wplus

def save_training_data(wminus, wplus, file_name) :
    np.save(file_name,np.stack((wminus,wplus)))


if __name__ == "__main__":
    deeplfts = DeepLFTS( (50, 50, 50), (1, 1, 1), True)
    deeplfts.train_model("training_data", "model_test" )
    #torch.save(deeplfts.net, 'initial.pt')
