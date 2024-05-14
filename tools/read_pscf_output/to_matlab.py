import sys
import string
import numpy as np
from scipy.io import savemat, loadmat

# Read omega and rho file
fp_o = open("omega", 'r')
fp_r = open("rho", 'r')
lines_omega = fp_o.readlines()
lines_rho = fp_r.readlines()

# Read parameters
params = {}
for i, line in enumerate(lines_omega):
    if "format" in line:
        params["format"] = lines_omega[i].split()[1:]
    elif "dim" in line:
        params["dim"] = int(lines_omega[i+1].split()[0])
    elif "crystal_system" in line:
        params["crystal_system"] = lines_omega[i+1].split()[0]
    elif "N_cell_param" in line:
        params["N_cell_param"] = int(lines_omega[i+1].split()[0])
    elif "cell_param" in line:
        params["cell_param"] = [float(f) for f in lines_omega[i+1].split()]
    elif "group_name" in line:
        params["group_name"] = lines_omega[i+1].split()[0]
    elif "N_monomer" in line:
        params["N_monomer"] = int(lines_omega[i+1].split()[0])
    elif "mesh" in line:
        params["mesh"] = [int(i) for i in lines_omega[i+1].split()]
        data_start = i+2
        break

if params["crystal_system"] == "cubic":
    params["cell_param"] = [params["cell_param"][0], params["cell_param"][0], params["cell_param"][0]]
elif params["crystal_system"] == "tetragonal":
    params["cell_param"] = [params["cell_param"][0], params["cell_param"][0], params["cell_param"][1]]
    
# Reverse order of mesh
params["mesh"].reverse()
params["cell_param"].reverse()

# Read data
w   = np.zeros([params["N_monomer"], np.prod(params["mesh"])])
phi = np.zeros([params["N_monomer"], np.prod(params["mesh"])])
for i, line in enumerate(lines_omega[data_start:]):
    w[:,i] = np.array([float(f) for f in line.split()])
for i, line in enumerate(lines_rho[data_start:]):
    phi[:,i] = np.array([float(f) for f in line.split()])

print(params)
print("len(lines_omega[data_start:]): ", len(lines_omega[data_start:]))
print("np.prod(params['mesh']): ", np.prod(params["mesh"]))
# print(w)
# print(phi)

# Make a dictionary for data
monomer_types = string.ascii_uppercase[:params["N_monomer"]]
mdic = {"dim":params["dim"],
    "crystal_system":params["crystal_system"], "N_cell_param":params["N_cell_param"],
    "group_name":params["group_name"], 
    "nx":params["mesh"], "lx":params["cell_param"],
    "chain_model":"continuous", "monomer_types":monomer_types}

# Add w and phi to the dictionary
for i, name in enumerate(monomer_types):
    mdic["w_" + name] = w[i]
    mdic["phi_" + name] = phi[i]

# Save data with matlab format
savemat("fields.mat", mdic, long_field_names=True, do_compression=True)
