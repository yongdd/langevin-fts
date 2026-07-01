import os
import numpy as np
from polymerfts import _core
from polymerfts import PlatformSelector

# -------------- initialize ------------

# OpenMP environment variables
os.environ["OMP_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation Parameters
f = 0.3                  # A-fraction, f
chi_n = 20               # Flory-Huggins Parameters * N
epsilon = 2.0            # a_A/a_B, conformational asymmetry
nx = [64, 64, 64]        # grids number
lx = [18., 6., 12.]      # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
ds = 1/100               # contour step interval
chain_model = "Continuous"  # choose among [Continuous, Discrete]

# calculate chain parameters, dict_a_n = [a_A, a_B]
dict_a_n = {"A": np.sqrt(epsilon*epsilon/(f*epsilon*epsilon + (1.0-f))),
            "B": np.sqrt(1.0/(f*epsilon*epsilon + (1.0-f)))}

# choose platform among [cuda, cpu-fftw, cpu-mkl]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)

# -------------- create instances (current API) ------------
factory = PlatformSelector.create_factory(platform, False)   # (platform, reduce_memory)

# molecules: (chain_model, ds, segment_lengths dict)
molecules = factory.create_molecules_information(chain_model, ds, dict_a_n)
# AB diblock with a grafting point ("G") at node 0 so we can inject a delta source
molecules.add_polymer(1.0, [["A", f, 0, 1], ["B", 1.0 - f, 1, 2]], {0: "G"})

optimizer = factory.create_propagator_computation_optimizer(molecules, True)
cb = factory.create_computation_box(nx, lx)
solver = factory.create_propagator_computation(cb, molecules, optimizer, "rqm4")

pc = molecules.get_polymer(0)
n_seg_total = pc.get_n_segment_total()
n_seg_A = pc.get_n_segment(0)   # segments in block 0 (A)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (cb.get_dim()))
print("chi_n: %f, f: %f, N: %d" % (chi_n, f, n_seg_total))
print("%s chain model" % (molecules.get_model_name()))
print("Conformational asymmetry (epsilon): %f" % (epsilon))
print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)))
print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)))
print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)))
print("Volume: %f" % (cb.get_volume()))

# -------------- allocate array ------------
total_grid = cb.get_total_grid()
w = {"A": np.zeros(total_grid, dtype=np.float64),
     "B": np.zeros(total_grid, dtype=np.float64)}

# point (delta) source at the origin for the grafting node "G"
q_init = {"G": np.zeros(total_grid, dtype=np.float64)}
q_init["G"][0] = np.prod(cb.get_nx()) / np.prod(cb.get_lx())

# squared distance from origin (cell-centered, periodic minimum image via folded index)
space_y, space_x, space_z = np.meshgrid(
    cb.get_lx(1)/cb.get_nx(1)*np.concatenate([np.arange((cb.get_nx(1)+1)//2), cb.get_nx(1)//2 - np.arange(cb.get_nx(1)//2)]),
    cb.get_lx(0)/cb.get_nx(0)*np.concatenate([np.arange((cb.get_nx(0)+1)//2), cb.get_nx(0)//2 - np.arange(cb.get_nx(0)//2)]),
    cb.get_lx(2)/cb.get_nx(2)*np.concatenate([np.arange((cb.get_nx(2)+1)//2), cb.get_nx(2)//2 - np.arange(cb.get_nx(2)//2)]))
squared_x = space_x**2 + space_y**2 + space_z**2

norm_segment = (f*epsilon**2 + (1 - f))

# -------------- compute propagators ------------
solver.compute_propagators(w, q_init)

# helper: global contour propagator q(r, n) for n = 0..n_seg_total
# block 0 (A): edge (0,1), local steps 0..n_seg_A
# block 1 (B): edge (1,2), local steps 0..(n_seg_total - n_seg_A)
def chain_propagator(n):
    if n <= n_seg_A:
        return solver.get_chain_propagator(0, 0, 1, n)
    return solver.get_chain_propagator(0, 1, 2, n - n_seg_A)

print("---------- Statistical Segment Length <x^2> ----------")
print("n'th segment, theory, calculation")

pred_mean_squared_x = 0.0
if molecules.get_model_name().lower() == "continuous":
    for n in range(0, n_seg_total + 1):
        q1_out = np.reshape(chain_propagator(n), cb.get_nx())
        mean_squared_x = np.sum(q1_out*squared_x)/np.sum(q1_out)

        print("%8d: %10.4f, %10.4f"
              % (n,
                 cb.get_dim()/3*pred_mean_squared_x,
                 n_seg_total*norm_segment*mean_squared_x))

        if n < n_seg_A:
            pred_mean_squared_x += epsilon**2
        else:
            pred_mean_squared_x += 1

elif molecules.get_model_name().lower() == "discrete":
    for n in range(1, n_seg_total + 1):
        q1_out = np.reshape(chain_propagator(n), cb.get_nx())
        mean_squared_x = np.sum(q1_out*squared_x)/np.sum(q1_out)

        print("%8d: %10.4f, %10.4f"
              % (n,
                 cb.get_dim()/3*pred_mean_squared_x,
                 n_seg_total*norm_segment*mean_squared_x))

        if n < n_seg_A:
            pred_mean_squared_x += epsilon**2
        elif n == n_seg_A:
            pred_mean_squared_x += (epsilon**2 + 1)/2
        else:
            pred_mean_squared_x += 1
