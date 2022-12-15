import sys
import numpy as np

#-------------- initialize ------------
print("Initializing")
np.set_printoptions(precision=10)
data_init = [
0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,
0.961913696e+0,0.792673860e-1,0.429684069e+0,0.290531312e+0,
0.453270921e+0,0.199228629e+0,0.754931905e-1,0.226924328e+0,
0.936407886e+0,0.979392715e+0,0.464957186e+0,0.742653949e+0,
0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0]
data_k_answer = [
10.6881904  +0j          , 0.7954998885+0.143345017j  ,-0.6668551075+0j,
0.4954041066+1.798776899j,-0.5050260775+0.04850456904j,-0.4504167737-1.947311157j,
0.5003159972-1.738407521j,-0.675930056 -0.09881542923j,-0.9823256426-0.6471590658j,
0.5003159972+1.738407521j,-0.6941925918+0.7499083742j ,-0.9823256426+0.6471590658j,
0.4954041066-1.798776899j,-1.659282438 +1.023353594j  ,-0.4504167737+1.947311157j]

data_init = np.reshape(data_init, (5,4))
data_k_answer = np.reshape(data_k_answer, (5,3))
#---------------- Forward --------------------
print("Running FFT 2D")
print("If error is less than 1.0e-7, it is ok!")
data_k = np.fft.rfftn(data_init)
error = np.max(np.absolute(data_k - data_k_answer))
print("FFT Forward Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)

#--------------- Backward --------------------
data_r = np.fft.irfftn(data_k_answer, (5,4))
error = np.max(np.absolute(data_r - data_init))
print("FFT Backward Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)

"""
#--------------- Test with large array --------------------
data_init = np.random.uniform(-1.0,1.0, size=(347,513))
data_k = np.fft.rfftn(data_init)
data_r = np.fft.irfftn(data_k, (data_init.shape))
error = np.max(np.absolute(data_r - data_init))
print("Test with lage array, Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)
"""
