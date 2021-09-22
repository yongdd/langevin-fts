import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from fts_dataset2d import *

#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
input_folder_name = "data2D/eval"
output_folder_name = "data2D/eval_diff"
model_file = "temp_models/FCN2d_5Layer_5kernel_128channel_epoch50.pth"
batch_size = 100

os.makedirs(output_folder_name, exist_ok=True)

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
net.eval()

file_list = glob.glob(input_folder_name + "/*.npz")
file_list.sort()
sample_data = np.load(file_list[0])
nx = sample_data['nx']
NN = sample_data['N']
print(sample_data.files)
#'nx', 'lx', 'N', 'f', 'chi_n', 'polymer_model', 'n_bar', 'random_seed', 'w_minus', 'w_plus'

with tqdm.tqdm(total=len(file_list), desc='Training Data', unit='data', leave=True) as pbar:
    for i, file_name in enumerate(file_list):
        data = np.load(file_name)
        w_plus = data['w_plus']
        w_minus = data['w_minus']
        w_minus_gpu = np.reshape(w_minus/NN, (1, 1, nx[0], nx[1]))
        w_minus_gpu = torch.tensor(w_minus_gpu, dtype=torch.float64).to(device)
        with torch.no_grad():
            w_plus_out = np.reshape(net(w_minus_gpu).cpu().numpy(), nx[0]*nx[1])*NN
        w_plus_diff = w_plus - w_plus_out
        pbar.update()
        
        np.savez(output_folder_name + "/fields_%06d.npz" % i,         
            nx = data['nx'],
            lx = data['lx'],
            N = data['N'],
            f = data['f'],
            chi_n = data['chi_n'],
            polymer_model = data['polymer_model'],
            n_bar = data['n_bar'],
            random_seed = data['random_seed'],
            w_minus = data['w_minus'],
            w_plus = data['w_plus'],
            w_plus_diff = w_plus_diff)
        
        if i % 500 ==0 :
            print([np.min(w_plus), np.min(w_plus_out)])
            vmin = np.min([np.min(w_plus), np.min(w_plus_out)])
            vmax = np.max([np.max(w_plus), np.max(w_plus_out)])
        
            plt.figure()
            fig, axes = plt.subplots(2,2, figsize=(10,10))
            axes[0,0].axis("off")
            axes[0,1].axis("off")
            axes[1,0].axis("off")
            axes[1,1].axis("off")

            axes[0,0].imshow(np.reshape(w_minus,nx), cmap="jet")
            axes[0,1].imshow(np.reshape(w_plus,nx), vmin=vmin, vmax=vmax, cmap="jet")
            axes[1,0].imshow(np.reshape(w_plus_out,nx), vmin=vmin, vmax=vmax, cmap="jet")
            axes[1,1].imshow(np.reshape(w_plus-w_plus_out,nx), vmin=-1, vmax=1, cmap="jet")
            
            plt.subplots_adjust(left=0.01,bottom=0.01,
                                top=0.99,right=0.99,
                                wspace=0.01, hspace=0.01)
            plt.savefig('field_%05d.png' % (i))
            plt.close()
