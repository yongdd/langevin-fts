import os
import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
input_folder_name = "../data2D_64/train"
output_folder_name = "data2D_64/train_diff"

os.makedirs(output_folder_name, exist_ok=True)

#model_file = "checkpoints/CP_epoch100.pth"
model_file = "../FCN2d_5Layer_5kernel_128channel_epoch50.pth"
batch_size = 100

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')
logging.info(f'Current cuda device {torch.cuda.current_device()}')
logging.info(f'Count of using GPUs {torch.cuda.device_count()}')

net = torch.load(model_file, map_location=device)
net.to(device=device)
logging.info(f'Model loaded from {model_file}')
net.double()

file_list = glob.glob(input_folder_name + "/*.npz")
file_list.sort()
sample_data = np.load(file_list[0])
print(sample_data.files)
nx = sample_data["nx"]

#'nx', 'lx', 'N', 'f', 'chi_n', 'polymer_model', 'n_bar', 'random_seed', 'w_minus', 'w_plus', 'phi_a', 'phi_b'

net.eval()
with tqdm.tqdm(total=len(file_list), desc='Data', unit='batch', leave=False) as pbar:
    for i, file_name in enumerate(file_list):
        #print(os.path.basename(file_name))
        data = np.load(file_name)
        w_minus = data['w_minus']
        w_plus = data['w_plus']
        
        w_minus_gpu = np.reshape(w_minus/10, (1, 1, nx[0], nx[1]))
        w_minus_gpu = torch.tensor(w_minus_gpu, dtype=torch.float64).to(device)
        with torch.no_grad():
            w_plus_gen = np.reshape(net(w_minus_gpu).cpu().numpy()*10, nx[0]*nx[1])
        w_plus_diff = w_plus-w_plus_gen
            
        out_file_name = os.path.join(output_folder_name, os.path.basename(file_name))
        np.savez(out_file_name, 
            nx=data['nx'], lx=data['lx'],
            N=data['N'], f=data['f'],
            chi_n=data['chi_n'],
            polymer_model=data['polymer_model'],
            n_bar=data['n_bar'],
            random_seed=data['random_seed'],
            w_minus=data['w_minus'], w_plus=data['w_plus'], w_plus_diff=w_plus_diff,
            phi_a=data['phi_a'], phi_b=data['phi_b'],          
        )
       
        
        if( i % 200 == 0):
            print([np.min(w_plus), np.min(w_plus_gen)])
            vmin = np.min([np.min(w_plus), np.min(w_plus_gen)])
            vmax = np.max([np.max(w_plus), np.max(w_plus_gen)])
            
            fig, axes = plt.subplots(2,2, figsize=(10,10))
            axes[0,0].axis("off")
            axes[0,1].axis("off")
            axes[1,0].axis("off")
            axes[1,1].axis("off")
             
            axes[0,0].imshow(np.reshape(w_minus,(nx[0], nx[1])), cmap="jet")
            axes[0,1].imshow(np.reshape(w_plus,(nx[0], nx[1])), vmin=vmin, vmax=vmax, cmap="jet")
            axes[1,0].imshow(np.reshape(w_plus_gen,(nx[0], nx[1])), vmin=vmin, vmax=vmax, cmap="jet")
            axes[1,1].imshow(np.reshape(w_plus_diff,(nx[0], nx[1])), vmin=-1, vmax=1, cmap="jet")
               
            plt.subplots_adjust(left=0.01,bottom=0.01,
                                top=0.99,right=0.99,
                                wspace=0.01, hspace=0.01)
            plt.savefig('w_plus_minus_%06d.png' % i)
            plt.close()
        
            pbar.update(200)

