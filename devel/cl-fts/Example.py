import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import clfts

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

f = 0.5        # A-fraction of major BCP chain, f
eps = 1.0      # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[32, 32, 32],          # Simulation grid numbers
    "lx":[4.209, 4.209, 4.209], # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B":12.4243},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
        
    "langevin":{                # Langevin Dynamics
        "max_step":500000,      # Langevin steps for simulation
        "dt":9.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":100000,          # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :100,      # Maximum number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "zeta_n" : 1/0.011111,    # Compressibility

    "compressor":{
        # "name":"am",                # Anderson Mixing
        # "name":"lr",                # Linear Response
        "name":"lram",              # Linear Response + Anderson Mixing
        "max_hist":20,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,             # Minimum mixing rate of simple mixing
        "mix_init":0.01,            # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each langevin step.
                            # 2 : Print at each saddle point iteration.
}
def read_complex_file(filepath, num_header_lines=3):
    header = []
    complex_data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                if i < num_header_lines:
                    header.append([float(x) for x in line.split()])
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        real, imag = map(float, parts)
                        complex_data.append(complex(real, imag))
                    else:
                        print(f"Warning: Unexpected format on line {i + 1}: {line}")
        
        # print("Header:")
        # for line in header:
        #     print(line)
        
        # print("\nComplex Data Preview:")
        # for z in complex_data[:5]:  # Show first 5 complex numbers
        #     print(z)
        
        return header, np.array(complex_data)

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    file_path = "input"  # replace with your actual filename
    header, data = read_complex_file(file_path)

    # print(header)
    print(data.shape)

    # fscanf(in,"%d %d %lf %lf %lf %lf",&N,&NA,&chi,&kappa,&C,&dt);
    # fscanf(in,"%d %d %d %lf %lf %lf",&m[0],&m[1],&m[2],&L[0],&L[1],&L[2]);
    # fscanf(in,"%d %d %d %d %lf %lf",&equil_its,&sim_its,&sample_freq,&write_freq,&K_t,&dK_t);

    # 90 45 11.829000 0.011111 316.227766 0.100000
    # 32 32 32 4.209000 4.209000 4.209000
    # 10000 400000 50 100000 1.800000 0.000400

# # Set random seed
# # If you want to obtain different results for each execution, set random_seed=None
# random_seed = 12345
# np.random.seed(random_seed)

# # Set initial fields
# input_data = loadmat("LamellaInput.mat", squeeze_me=True)
# w_A = input_data["w_A"]
# w_B = input_data["w_B"]

# # Initialize calculation
# simulation = clfts.CLFTS(params=params, random_seed=random_seed)

# # Set a timer
# time_start = time.time()

# # # Continue simulation with recorded field configurations and random state.
# # simulation.continue_run(file_name="fields_010000.mat")

# # Run
# simulation.run(initial_fields={"A": w_A, "B": w_B})