################################################################################
# WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
# DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
# - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
# - NEVER decrease field strength or standard deviation values
# - NEVER change grid sizes, box dimensions, or polymer parameters
# - NEVER weaken any test conditions to make tests pass
# These parameters are carefully calibrated. If a test fails, report the
# failure to the user rather than modifying the test to pass.
################################################################################

import sys
import numpy as np

#-------------- initialize ------------
print("Initializing")
np.set_printoptions(precision=10)
data_init = [0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0]
data_k_answer = [3.497839818+0j, -0.7248383037+0.4777381112j, -0.5654030903-0.05431399883j]

#---------------- Forward --------------------
print("Running FFT 1D")
print("If error is less than 1.0e-7, it is ok!")
data_k = np.fft.rfft(data_init)
error = np.max(np.absolute(data_k - data_k_answer))
print("FFT Forward Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)

#--------------- Backward --------------------
data_r = np.fft.irfft(data_k_answer, len(data_init))
error = np.max(np.absolute(data_r - data_init))
print("FFT Backward Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)

"""
#--------------- Test with large array --------------------
data_init = np.random.uniform(-1.0,1.0, 5797)
data_k = np.fft.rfft(data_init)
data_r = np.fft.irfft(data_k, len(data_init))
error = np.max(np.absolute(data_r - data_init))
print("Test with lage array, Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)
"""
