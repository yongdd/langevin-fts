import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

zeta_n = 100

mdic_ref = loadmat("gyroid_ref.mat", squeeze_me=True)
nx = mdic_ref['nx']
lx = mdic_ref['lx']

for name in ["morse", "matsen"]: #, , "simple"

    print("name:", name)

    mdic = loadmat("gyroid_%s_%d.mat" % (name, zeta_n), squeeze_me=True)

    phi_a_diff = mdic_ref['phi_A'] - mdic['phi_A']
    phi_b_diff = mdic_ref['phi_B'] - mdic['phi_B']

    w_a_diff = mdic_ref['w_A'] - mdic['w_A']
    w_b_diff = mdic_ref['w_B'] - mdic['w_B']

    w_p_ref = (mdic_ref['w_A'] + mdic_ref['w_B'])/2
    w_n_ref = (mdic_ref['w_A'] - mdic_ref['w_B'])/2

    w_p = (mdic['w_A'] + mdic['w_B'])/2
    w_n = (mdic['w_A'] - mdic['w_B'])/2

    w_p_diff = w_p_ref - w_p
    w_m_diff = w_n_ref - w_n

    # w_p_diff = (w_a_diff + w_b_diff)/2
    # w_m_diff = (w_a_diff - w_b_diff)/2

    # print("std_phi_a_diff: ", np.std(phi_a_diff))
    # print("std_phi_b_diff: ", np.std(phi_a_diff))
    # print("std_w_a_diff: ", np.std(w_a_diff))
    # print("std_w_b_diff: ", np.std(w_b_diff))
    print("std_w_p_diff: ", np.std(w_p_diff))
    print("std_w_m_diff: ", np.std(w_m_diff))

    # print("mean_phi_a_diff: ", np.mean(phi_a_diff))
    # print("mean_phi_b_diff: ", np.mean(phi_a_diff))
    # print("mean_w_a_diff: ", np.mean(w_a_diff))
    # print("mean_w_b_diff: ", np.mean(w_b_diff))
    print("mean_w_p_diff: ", np.mean(w_p_diff))
    print("mean_w_m_diff: ", np.mean(w_m_diff))

# mdic = loadmat("disorder_ref.mat" , squeeze_me=True)
mdic = loadmat("disorder_morse_10.mat" , squeeze_me=True)

print(mdic['w_A'])
print(mdic['w_B'])

print((mdic['w_A']+mdic['w_B'])/2)
print((mdic['w_A']-mdic['w_B'])/2)