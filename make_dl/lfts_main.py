import time
import matplotlib.pyplot as plt
from pathlib import Path
from lfts_fortran import lfts
from lfts_torch import *

# Initialize L-FTS modules
lfts.initialize("inputs")

# Read boundary conditions
bc_x = lfts.get_param_int_idx("geometry.bc",1)[1]
bc_y = lfts.get_param_int_idx("geometry.bc",2)[1]
bc_z = lfts.get_param_int_idx("geometry.bc",3)[1]

# Read deep learning parameters 
chi_n = lfts.get_chin()

start_record_data = lfts.get_param_int("dl.start_record_data")[1]
end_record_data   = lfts.get_param_int("dl.end_record_data")[1]
data_record_period = lfts.get_param_int("dl.record_period")[1]
training_data_folder = lfts.get_param_str("dl.training_data_folder")[1].strip().decode('utf-8')
#training_data_folder += "_%06d" % np.round(1000*chi_n)
Path(training_data_folder).mkdir(parents=True, exist_ok=True)
#Path("image_folder").mkdir(parents=True, exist_ok=True)

train_model         =  True if (lfts.get_param_int("dl.train_model")[1]          == 1) else False
use_pretrained_model = True if (lfts.get_param_int("dl.use_pretrained_model")[1] == 1) else False
pretrained_model = lfts.get_param_str("dl.load_model_file")[1].strip().decode('utf-8')
if ( not train_model and not use_pretrained_model):
    print("You need a pretrained_model or have to train new model")

# Check data types
print(type(lfts.wminus[0,0,0]))

process_idx = lfts.get_process_idx()
print("process_idx: ", process_idx)

# Initialize deep learning module

if ( use_pretrained_model ) :
    deeplfts = DeepLFTS(lfts.wminus.shape, (bc_x, bc_y, bc_z), train_model, model_file=pretrained_model)
else :
    deeplfts = DeepLFTS(lfts.wminus.shape, (bc_x, bc_y, bc_z), train_model)

# Record start time
start_time = time.time()

# Langevin iteration begins here
while (lfts.langevin_iter <= lfts.langevin_end_iter):
    lfts.pre_langevin()
    if ( deeplfts.model_is_trained ):
        lfts.wplus = deeplfts.predict(lfts.wminus)

    #plt.figure()
    #plt.subplot(1,3,1)
    #plt.imshow(lfts.wminus[:,:,25])
    #plt.subplot(1,3,2)
    #plt.imshow(lfts.wplus[:,:,25])
    lfts.find_saddle_point()

    #plt.subplot(1,3,3)
    #plt.imshow(lfts.wplus[:,:,25])
    #plt.savefig( "./image_folder/%06d_%06d.png" % (np.round(1000*chi_n), lfts.langevin_iter))
    #plt.close()

    if ( start_record_data <= lfts.langevin_iter and 
    lfts.langevin_iter <= end_record_data and
    (lfts.langevin_iter - start_record_data) % data_record_period == 0):
        save_file_name = training_data_folder + '/fields_%06d_%06d' % (np.round(1000*chi_n), lfts.langevin_iter)
        save_training_data(lfts.wminus,lfts.wplus,save_file_name)
    if ( lfts.langevin_iter > end_record_data and not deeplfts.model_is_trained):
        print("Synchronizing before training, proc_idx: ", process_idx)
        lfts.parallel_barrier()
        if (process_idx == 0 ) :
            deeplfts.train_model(training_data_folder, "model_%06d" % (lfts.langevin_iter))
        lfts.parallel_barrier()
        print("Synchronizing after training, proc_idx: ", process_idx)
        if (process_idx != 0 ) :
            print("Load trained models, proc_idx: ", process_idx)
            deeplfts.load_model_weights( "model_%06d.pt" % (lfts.langevin_iter) )
    lfts.post_langevin()

# Print total simulation time
end_time = time.time()
total_time = end_time - start_time
print ("total time: ", total_time)
print ("time per langevin step: ", total_time/(lfts.langevin_end_iter-lfts.langevin_start_iter+1) )
lfts.write_final_output()
lfts.finalize()
