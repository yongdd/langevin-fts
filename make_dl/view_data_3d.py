import matplotlib.pyplot as plt
import numpy as np


file_name = 'training_data/fields_020900_002002.npy'
fields = np.load(file_name)

fields = fields.reshape(2, 50, 50, 50)

plt.subplot(1, 2, 1)
plt.imshow(fields[0,:,:,25])
plt.subplot(1, 2, 2)
plt.imshow(fields[1,:,:,25])
plt.show()

print(np.mean(fields[0,:,:,:]))
print(np.mean(fields[1,:,:,:]))
print(np.std(fields[0,:,:,:]))
print(np.std(fields[1,:,:,:]))

#plt.imshow(Y_diff[10,:,:,25], vmin=-1, vmax=1, cmap=plt.get_cmap('PiYG'))
#plt.colorbar()
#plt.show()

#plt.imshow(Y_diff[100,:,:,0]/(np.abs(Y_test[100,:,:,0])+1), vmin=-0.1, vmax=0.1, cmap=plt.get_cmap('PiYG'))
#plt.colorbar()
#plt.show()
