from lfts_pytorch import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    model_name = 'model_002001_00.pt'
    file_name = 'training_data_018000/fields_018000_001500.npy' 
    deeplfts = DeepLFTS( (50, 50, 50), (1, 1, 1), False, model_name)
    fields = np.load(file_name)
    wminus = fields[0,:,:,:]
    wplus = deeplfts.predict(wminus)

    plt.figure()
    plt.imshow(fields[0,:,:,25])
    plt.colorbar()
    plt.savefig('1test_1wminus.png')

    plt.figure()
    plt.imshow(fields[1,:,:,25])
    plt.colorbar()
    plt.savefig('1test_2wplus.png')

    plt.figure()
    plt.imshow(wplus[:,:,25])
    plt.colorbar()
    plt.savefig('1test_3predict.png')
