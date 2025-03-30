import sys
import numpy as np

#-------------- initialize ------------
print("Initializing")
np.set_printoptions(precision=10)

data_init = [ 0.2733181541-0.0187024081j, -0.1507659497+0.6894673618j,
             -0.0350475991+0.2143790085j, -0.0175411299+0.4013954178j,
              0.8336831753+0.4648317909j]

data_k_answer = [ 0.9036466507+1.7513711709j,  0.6306119942+0.7863804708j,
                  0.0144763523-0.2002702411j, -0.6053249486-1.3242603125j,
                  0.4231807219-1.1067331286j]

# for i in range(len(data_init)):
#     print(f"{{{data_init[i].real},{data_init[i].imag}}}, " , end="")
#     if i % 2 ==1:
#         print("")

# print("")
# for i in range(len(data_k_answer)):
#     print(f"{{{data_k_answer[i].real},{data_k_answer[i].imag}}}, " , end="")
#     if i % 2 ==1:
#         print("")

#---------------- Forward --------------------
print("Running FFT 1D")
print("If error is less than 1.0e-7, it is ok!")
data_k = np.fft.fft(data_init)
error = np.max(np.absolute(data_k - data_k_answer))
print("FFT Forward Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)

#--------------- Backward --------------------
data_r = np.fft.ifft(data_k_answer)
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
