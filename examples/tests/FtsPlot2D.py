import numpy as np
import matplotlib.pyplot as plt

data = np.load("field_update_algorithms/fields_EM1.npz")
print(data.files)
w_a = np.reshape(data['w'][0], data['nx'])
w_b = np.reshape(data['w'][1], data['nx'])

print(w_a.shape)

fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].axis("off")
axes[0,1].axis("off")
axes[1,0].axis("off")
axes[1,1].axis("off")

axes[0,0].imshow(w_a[0,:,:], cmap="jet")
axes[0,1].imshow(w_b[0,:,:], cmap="jet")

plt.subplots_adjust(left=0.01,bottom=0.01,
                    top=0.99,right=0.99,
                    wspace=0.01, hspace=0.01)
plt.savefig('field.png')
