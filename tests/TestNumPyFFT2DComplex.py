import sys
import numpy as np

#-------------- initialize ------------
print("Initializing")
np.set_printoptions(precision=10)

data_init = [
 0.7479851438+0.5113923184j, -0.611924137 +0.6688134066j,
-0.2562048442+0.7156670162j, -0.698469607 +0.3809731847j,
 0.5071785071-0.8416524347j, -0.1415995644+0.9279043295j,
 0.1582071603-0.2578680913j, -0.9819601031+0.4541845031j,
 0.2625178548-0.7306389533j,  0.0197493655+0.6360389997j,
-0.6824309284-0.0573651711j, -0.588476528 -0.4983797445j,
 0.6192052822-0.3934574836j,  0.582460527 +0.9074145996j,
 0.4822692317-0.4687287157j, -0.6937104783-0.9072499747j,
-0.7862444092+0.3346206151j, -0.7524654201+0.0442255985j,
 0.420599632 -0.7957530045j,  0.9435906198+0.9249318616j]

data_k_answer = [
-1.4497226956+1.5550728596j,  4.0581392311-1.3709348392j,
 4.3958879555-5.5226406686j, -1.6017349765+0.8595588959j,
-1.1054099382+5.1776018019j, -6.4498665458-0.6920906220j,
 0.0443508843-1.1905937191j,  2.0370359645-2.3202722334j,
-0.6401006748-0.5453888601j,  1.1391904384-3.7814794759j,
 0.6683255063+0.3989231452j,  2.3346020000+1.8269050208j,
 0.0275726665+2.8851798905j,  5.9827303722-0.6033125243j,
 1.7837980751+5.1801546701j, -1.3263453051-0.7671836679j,
-0.9254065801+2.3117639377j,  1.7299575537+4.9937166227j,
 2.1185077968+2.0205202883j,  2.1381911477-0.1876541545j]

# for i in range(len(data_init)):
#     print(f"{{{data_init[i].real},{data_init[i].imag}}}, " , end="")
#     if i % 2 ==1:
#         print("")

# print("")
# for i in range(len(data_k_answer)):
#     print(f"{{{data_k_answer[i].real},{data_k_answer[i].imag}}}, " , end="")
#     if i % 2 ==1:
#         print("")

data_init = np.reshape(data_init, (5,4))
data_k_answer = np.reshape(data_k_answer, (5,4))
#---------------- Forward --------------------
print("Running FFT 2D")
print("If error is less than 1.0e-7, it is ok!")
data_k = np.fft.fftn(data_init)
error = np.max(np.absolute(data_k - data_k_answer))
print("FFT Forward Error: ", error)
if(np.isnan(error) or error > 1e-7):
    sys.exit(-1)

#--------------- Backward --------------------
data_r = np.fft.ifftn(data_k_answer)
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
