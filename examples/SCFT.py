import sys
import numpy as np
from langevinfts import *

# -------------- initialize ------------
# read file name
print(sys.argv[0])
if(len(sys.argv) < 2):
    print("Input parameter file is required, e.g, 'scft.out inputs'")
    sys.exit()

# initialize ParamParser
pp = ParamParser.get_instance()
pp.read_param_file(sys.argv[1])

# choose platform
str_platform = pp.get("platform")
if (len(str_platform)==0):
    factory = KernelFactory()
else:
    factory = KernelFactory(str_platform[0])

# read simulation box parameters
nx = pp.get("geometry.grids")
lx = pp.get("geometry.box_size")

# read chain parameters

print(pp.get("chain.a_fraction")[0])

f = float(pp.get("chain.a_fraction")[0])
NN = int(pp.get("chain.contour_step")[0])
chi_n = float(pp.get("chain.chi_n")[0])

if(len(nx) != 3):
    print("geometry.grids is not specified.")
    sys.exit()
if(len(lx) != 3):
    print("geometry.box_size is not specified.")
    sys.exit()
nx = [int(i) for i in nx]
lx = [float(f) for f in lx]

# read Anderson mixing parameters
# anderson mixing begin if error level becomes less then start_anderson_error
am_start = float(pp.get("am.start_error")[0])
# max number of previous steps to calculate new field
max_anderson = float(pp.get("am.step_max")[0])
# minimum mixing parameter
mix_min = float(pp.get("am.mix_min")[0])
# initial mixing parameter
mix_init = float(pp.get("am.mix_init")[0])

# read iteration parameters
tolerance = float(pp.get("iter.tolerance")[0])
maxiter = int(pp.get("iter.step_saddle")[0])

# create instances and assign to the variables of base classs
# for the dynamic binding
pc = factory.create_polymer_chain(f, NN, chi_n)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am = factory.create_anderson_mixing(sb, 2, max_anderson, am_start, mix_min, mix_init)

#print(dir(pp))
#print(dir(factory))
#print(dir(sb))

# assign large initial value for the energy and error
energy_total = 1.0e20;
error_level = 1.0e20;

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: 3")
print("Precision: 8")
print("chi_n: %f, f: %f, NN: %d" % (pc.chi_n, pc.f, pc.NN) )
print("Nx: %d, %d, %d" % (sb.nx[0], sb.nx[1], sb.nx[2]) )
print("Lx: %f, %f, %f" % (sb.lx[0], sb.lx[1], sb.lx[2]) )
print("dx: %f, %f, %f" % (sb.dx[0], sb.dx[1], sb.dx[2]) )

total = 0.0;

for i in range(0, sb.MM):
    total += sb.dv_at(i);
print("volume: %f, sum(dv): %f" % (sb.volume, total) )

#-------------- allocate array ------------
w       = np.zeros([2, sb.MM], dtype=np.float64)
w_out   = np.zeros([2, sb.MM], dtype=np.float64)
w_diff  = np.zeros([2, sb.MM], dtype=np.float64)
xi      = np.zeros( sb.MM,     dtype=np.float64)
phia    = np.zeros( sb.MM,     dtype=np.float64)
phib    = np.zeros( sb.MM,     dtype=np.float64)
phitot  = np.zeros( sb.MM,     dtype=np.float64)
w_plus  = np.zeros( sb.MM,     dtype=np.float64)
w_minus = np.zeros( sb.MM,     dtype=np.float64)
q1_init = np.zeros( sb.MM,     dtype=np.float64)
q2_init = np.zeros( sb.MM,     dtype=np.float64)

print("wminus and wplus are initialized to a given test fields.")
for i in range(0,sb.nx[0]):
    for j in range(0,sb.nx[1]):
        for k in range(0,sb.nx[2]):
            idx = i*sb.nx[1]*sb.nx[2] + j*sb.nx[2] + k;
            phia[idx]= np.cos(2.0*np.pi*i/4.68)*np.cos(2.0*np.pi*j/3.48)*np.cos(2.0*np.pi*k/2.74)*0.1;

for i in range(0,sb.MM):
    phib[i] = 1.0 - phia[i];
    w[0, i] = chi_n*phib[i];
    w[1, i] = chi_n*phia[i];

# keep the level of field value
sb.zero_mean(w[0]);
sb.zero_mean(w[1]);

# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
for i in range(0,sb.MM):
    q1_init[i] = 1.0;
    q2_init[i] = 1.0;
