"""Fourier transforms and Fourier multipliers of whole fields: PropagatorSolver.fourier.

The Fourier transform uses the FFT of the platform (FFTW or MKL on the CPU, cuFFT
on the GPU). Besides plain transforms it applies linear operators that are diagonal
in k, such as smearing kernels, Green's functions or the exact step of a linear
Langevin equation. This script shows each function and checks it against NumPy.

    forward(fields)              fields -> Fourier coefficients (transform only)
    backward(coefficients)       Fourier coefficients -> fields (inverse transform only)
    k_squared()                  |k|^2 of every coefficient, a new NumPy array
    set_multiplier(M)            stores M(k) and returns a handle (no transform)
    apply([(h, fields), ...])    fields -> FFT -> sum of M(k) * coefficients -> inverse FFT

Fields are flat NumPy arrays as everywhere in polymerfts: (n_grid,) for one field,
(n_fields, n_grid) for several. Inputs and results are NumPy arrays on every platform;
with CUDA only the stored multipliers and the work inside a call live on the GPU.
"""

import numpy as np
from polymerfts import PropagatorSolver

nx, lx = [64, 48], [4.0, 3.0]
n_grid = nx[0] * nx[1]

# The transform needs only the grid and the box; no polymers have to be added
# (changing the box with set_lx or set_lattice_parameters does need them).
# "auto" takes CUDA when it is available (2D and 3D), otherwise the CPU.
solver = PropagatorSolver(nx=nx, lx=lx, ds=0.01, bond_lengths={"A": 1.0}, bc=["periodic"] * 4,
                          chain_model="continuous", platform="auto")
ft = solver.fourier
print(f"Platform: {solver.platform}\n")

rng = np.random.default_rng(1)
w = rng.normal(0.0, 5.0, n_grid)             # one field
ws = rng.normal(0.0, 5.0, (2, n_grid))       # two fields


def compare(name, result, reference):
    """Print the largest difference from the NumPy reference and stop if it is not rounding."""
    difference = np.max(np.abs(result - reference))
    print(f"{name:<52s} max |difference| = {difference:.1e}")
    assert difference <= 1e-10 * np.max(np.abs(reference)), name


# ---------------------------------------------------------------------------
# 1. Fourier transform only: forward() and backward()
# ---------------------------------------------------------------------------
# Real fields (SCFT, L-FTS) give the half spectrum of numpy.fft.rfftn: the last axis
# keeps the frequencies 0..n/2, the others follow from x(-k) = conj(x(k)). Complex fields
# (field_type="complex", CL-FTS) give the full spectrum of numpy.fft.fftn (see part 6).
# As in NumPy, forward() has no 1/N and backward() includes it.
w_k = ft.forward(w)                          # complex, shape (64, 25)
print(f"coefficient shape {ft.coefficient_shape}, forward(w).shape {w_k.shape}")
compare("forward(w) vs numpy.fft.rfftn", w_k, np.fft.rfftn(w.reshape(nx)))
compare("backward(forward(w)) vs w", ft.backward(w_k), w)

ws_k = ft.forward(ws)                        # several fields at once: shape (2, 64, 25)
compare("forward of two fields", ws_k, np.fft.rfftn(ws.reshape(2, *nx), axes=(1, 2)))

# ---------------------------------------------------------------------------
# 2. |k|^2 of every coefficient: k_squared()
# ---------------------------------------------------------------------------
# k_squared() returns a new NumPy array of the coefficient shape: k2[i, j] is |k|^2 of
# the coefficient w_k[i, j]. In this orthogonal box k = (2 pi m_x / L_x, 2 pi m_y / L_y)
# with the signed integer frequencies m of numpy.fft.fftfreq. Oblique cells use the
# reciprocal metric with the shortest alias of each mode (as the propagator solvers);
# walls use the cosine series with k = pi m / L. The array is computed from the current
# box: after a box change (set_lx) call k_squared() again and store the multipliers again.
k2 = ft.k_squared()
kx = 2 * np.pi * np.fft.fftfreq(nx[0], d=lx[0] / nx[0])
ky = 2 * np.pi * np.fft.rfftfreq(nx[1], d=lx[1] / nx[1])
compare("k_squared() vs (2 pi m / L)^2", k2, kx[:, None] ** 2 + ky[None, :] ** 2)

# ---------------------------------------------------------------------------
# 3. A multiplier on several fields: Gaussian smearing
# ---------------------------------------------------------------------------
# set_multiplier(M) does not transform anything: it checks M, stores it (copies it
# to the GPU with CUDA) and returns a handle. M is a function of k, made from k2.
# The transforms happen in apply():  apply([(h, fields)]) = backward(M * forward(fields)).
# A multiplier of the coefficient shape is applied to every field of its term. With real
# fields M acts as a real operator, M(-k) = conj(M(k)): on the planes of the half spectrum
# that hold both k and -k the transform replaces M by (M(k) + conj(M(-k)))/2.
a = 0.2
gamma = np.exp(-a ** 2 * k2 / 2)             # Gamma(k) of Gaussian smearing
h_gamma = ft.set_multiplier(gamma)           # store once ...
w_smeared = ft.apply([(h_gamma, ws)])        # ... apply as often as needed: shape (2, n_grid)
reference = np.fft.irfftn(gamma * np.fft.rfftn(ws.reshape(2, *nx), axes=(1, 2)), s=nx, axes=(1, 2))
compare("apply: smearing vs NumPy", w_smeared, reference.reshape(2, n_grid))

# The same result in three steps. apply() is faster on the GPU: the coefficients stay
# there instead of going to the host for the product and back for the inverse transform.
compare("apply vs backward(gamma * forward(ws))", w_smeared, ft.backward(gamma * ft.forward(ws)))

# ---------------------------------------------------------------------------
# 4. Several terms in one call: exact step of a linear Langevin equation
# ---------------------------------------------------------------------------
# Each Fourier mode of  dw/dt = laplacian(w) + noise  relaxes as exp(-k^2 t). Over a
# time step dt its exact solution is
#     w(k) -> exp(-k^2 dt) w(k) + S(k) xi(k),    S(k)^2 = (1 - exp(-2 k^2 dt)) / k^2,
# where xi is a field of independent standard normal values and S(0)^2 = 2 dt (the noise
# here has variance 2 dt per grid value and unit time). apply() takes a list of
# (handle, fields) terms, sums M(k) * forward(fields) over the terms in k space and
# transforms back once:
#     apply([(h1, f1), (h2, f2)]) = backward(M1 * forward(f1) + M2 * forward(f2))
dt = 0.01
decay = np.exp(-k2 * dt)
noise_amplitude = np.sqrt(np.divide(1 - np.exp(-2 * k2 * dt), k2, out=np.full_like(k2, 2 * dt), where=k2 > 0))
h_decay = ft.set_multiplier(decay)
h_noise = ft.set_multiplier(noise_amplitude)
xi = rng.normal(0.0, 1.0, n_grid)
w_next = ft.apply([(h_decay, w), (h_noise, xi)])   # one flat field in, one flat field out
reference = np.fft.irfftn(decay * np.fft.rfftn(w.reshape(nx)) + noise_amplitude * np.fft.rfftn(xi.reshape(nx)),
                          s=nx, axes=(0, 1))
compare("apply: two terms vs NumPy", w_next, reference.ravel())

# ---------------------------------------------------------------------------
# 5. Matrix multipliers: several inputs and outputs
# ---------------------------------------------------------------------------
# A multiplier of shape (n_out, n_in) + coefficient shape maps n_in fields to n_out
# fields:  out_i = backward(sum_j M_ij(k) * forward(in_j)).  Example: the screened
# electrostatic potential of two charged species with valences z,
#     psi(k) = (z_A phi_A(k) + z_B phi_B(k)) / (k^2 + kappa^2)      (n_out = 1, n_in = 2).
# The etd integrator of CL-FTS uses (m, m) matrices of this kind for m coupled fields.
# Every term of one apply() call must give the same number of output fields.
z, kappa_sq = np.array([1.0, -1.0]), 4.0
green = z[None, :, None, None] / (k2 + kappa_sq)    # shape (1, 2, 64, 25)
h_green = ft.set_multiplier(green)
phi = 1.0 + 0.1 * rng.normal(size=(2, n_grid))      # two concentration fields
psi = ft.apply([(h_green, phi)])                    # shape (1, n_grid)
phi_k = np.fft.rfftn(phi.reshape(2, *nx), axes=(1, 2))
reference = np.fft.irfftn((z[0] * phi_k[0] + z[1] * phi_k[1]) / (k2 + kappa_sq), s=nx, axes=(0, 1))
compare("apply: matrix multiplier vs NumPy", psi[0], reference.ravel())

# ---------------------------------------------------------------------------
# 6. Complex fields (CL-FTS)
# ---------------------------------------------------------------------------
# With field_type="complex" the coefficients are the full spectrum of numpy.fft.fftn,
# k_squared() has the full grid shape and multipliers may be any complex functions of k.
# Real arrays passed as fields (e.g. real noise) are converted to complex.
solver_c = PropagatorSolver(nx=nx, lx=lx, ds=0.01, bond_lengths={"A": 1.0}, bc=["periodic"] * 4,
                            chain_model="continuous", platform="auto", field_type="complex")
ft_c = solver_c.fourier
wc = rng.normal(size=n_grid) + 1j * rng.normal(size=n_grid)
print(f"\ncomplex fields: coefficient shape {ft_c.coefficient_shape}")
compare("forward(complex w) vs numpy.fft.fftn", ft_c.forward(wc), np.fft.fftn(wc.reshape(nx)))
shift = np.exp(-1j * 2 * np.pi * np.fft.fftfreq(nx[0])[:, None] * 3)   # translation by 3 grid points along x
h_shift = ft_c.set_multiplier(np.broadcast_to(shift, ft_c.coefficient_shape))
compare("apply: complex multiplier (translation)", ft_c.apply([(h_shift, wc)]), np.roll(wc.reshape(nx), 3, axis=0).ravel())
