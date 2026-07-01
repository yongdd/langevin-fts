"""
Verify DTFT-based bond function (Park et al., JCP 2019, Appendix / Eq. 30)
for the SPRING-BEAD (BS / Gaussian) model, NOT FJC.

Park's Eq. (30) gives the cell-averaged bond function in Fourier space for FJC.
The paper states (p.11): "All we need to do is to replace the first sinc function
of Eq. (30) with the known Fourier transform of the bond function while leaving
the last three sinc functions."

For the spring-bead model the FT of the Gaussian bond function is (Eq. 22):
    g~(xi) = exp(-(2/3) pi^2 a^2 |xi|^2)

Conventions (Park):
  - FT:  f~(xi) = int f(r) exp(-2 pi i xi.r) dr           (Eq. 21)
  - grid points r_n = n*dx, n=0..I,  I = L/dx             (vertex-centered)
  - DFT (periodic) samples xi = i'/L  (i' = i or i-I)     (Eq. 28)
  - DCT (Neumann)  samples xi = i/(2L)                    (Eq. 27)
  - Probability:  sum_i g_i dV_i = 1                      (Eq. 24a)
  - RMS step:     sum_i r_i^2 g_i dV_i = a^2              (Eq. 24b)

Real-space Gaussian bond function (1D), with a^2 = b^2 ds:
    g(z) = sqrt(3/(2 pi a^2)) exp(-3 z^2 / (2 a^2))
         = Normal(0, sigma^2) with sigma^2 = a^2/3

We validate using the PERIODIC (DFT) formulation, which is what the discrete-chain
pseudo-spectral solver in this codebase uses (g~(k) = exp(-b^2 k^2 ds/6)).
"""
import numpy as np
from scipy.special import erf

# ---------------------------------------------------------------------------
# Continuous quantities
# ---------------------------------------------------------------------------
def gaussian_1d(z, a):
    """Real-space 1D spring-bead bond function g(z)."""
    s2 = a * a / 3.0
    return np.exp(-z * z / (2.0 * s2)) / np.sqrt(2.0 * np.pi * s2)

def gaussian_ft(xi, a):
    """FT of the Gaussian bond function, Eq. (22): exp(-(2/3) pi^2 a^2 xi^2)."""
    return np.exp(-(2.0 / 3.0) * np.pi**2 * a * a * xi * xi)

# ---------------------------------------------------------------------------
# Ground truth: analytic CELL-AVERAGE of the periodic Gaussian (Eq. 29)
# gbar_n = (1/dz) int_{z_n-dz/2}^{z_n+dz/2} g_per(z) dz   (periodic images summed)
# Gaussian integral -> erf, exact.
# ---------------------------------------------------------------------------
def cell_avg_periodic(I, L, a, n_img=20):
    dz = L / I
    sig = a / np.sqrt(3.0)
    n = np.arange(I)
    zc = n * dz
    g = np.zeros(I)
    for m in range(-n_img, n_img + 1):
        zp = zc + m * L
        hi = (zp + dz / 2.0) / (sig * np.sqrt(2.0))
        lo = (zp - dz / 2.0) / (sig * np.sqrt(2.0))
        g += 0.5 * (erf(hi) - erf(lo)) / dz
    return g

# ---------------------------------------------------------------------------
# DTFT method (periodic / DFT analog of Eq. 30) for spring-bead.
# g~_q = (1/dV) sum_m exp(-(2/3) pi^2 a^2 ((q-mI)/L)^2) * sinc((q-mI)/I)
# (first factor = Gaussian FT; sinc = FT of rectangular cell window)
# Then real-space g_n = IDFT(g~_q).
# ---------------------------------------------------------------------------
def bond_dtft_periodic(I, L, a, n_alias=10):
    dz = L / I
    q = np.arange(I)
    gt = np.zeros(I)
    for m in range(-n_alias, n_alias + 1):
        qa = q - m * I
        xi = qa / L
        gt += gaussian_ft(xi, a) * np.sinc(qa / I)   # np.sinc is normalized sinc
    gt = gt / dz                                     # 1/dV prefactor (dV=dz in 1D)
    g_real = np.fft.ifft(gt).real                    # g~_q are DFT coeffs of g_n
    return g_real, gt

# ---------------------------------------------------------------------------
# Naive real-space point sampling of the Gaussian (the simple/"wrong" way).
# g_n = g_per(z_n), then rescale to satisfy Eq. (24a) probability conservation.
# ---------------------------------------------------------------------------
def bond_naive_realspace(I, L, a, n_img=20, normalize=True):
    dz = L / I
    n = np.arange(I)
    zc = n * dz
    g = np.zeros(I)
    for m in range(-n_img, n_img + 1):
        g += gaussian_1d(zc + m * L, a)
    if normalize:
        g = g / (np.sum(g) * dz)   # enforce sum g dz = 1
    return g

# ---------------------------------------------------------------------------
# Naive Fourier sampling (what the discrete-chain solver currently uses):
# g~(k) = exp(-b^2 k^2 ds/6) = exp(-(2/3) pi^2 a^2 (q'/L)^2), no cell filter,
# no alias sum. (Eq. 22 / 28 for BS.)
# ---------------------------------------------------------------------------
def bond_naive_fourier(I, L, a):
    dz = L / I
    q = np.arange(I)
    qp = np.where(q < I // 2, q, q - I)   # i' = i or i-I
    xi = qp / L
    gt = gaussian_ft(xi, a) / dz
    g_real = np.fft.ifft(gt).real
    return g_real, gt

# ---------------------------------------------------------------------------
# Diagnostics: Eq. (24) constraints with periodic minimum-image distance.
# ---------------------------------------------------------------------------
def constraints(g, I, L):
    dz = L / I
    n = np.arange(I)
    z = n * dz
    z = np.where(z > L / 2, z - L, z)   # minimum image
    prob = np.sum(g) * dz               # Eq. 24a -> should be 1
    rms2 = np.sum(z * z * g) * dz       # Eq. 24b -> should be a^2
    return prob, rms2


def run_1d():
    a = 1.0
    L = 8.0 * a
    print("=" * 78)
    print(f"1D spring-bead bond function, a={a}, box L={L} (={L/a:.0f}a), target a^2={a*a}")
    print("=" * 78)
    print(f"{'I':>4} {'dz/a':>6} | {'method':<14} "
          f"{'prob(24a)':>11} {'rms^2(24b)':>11} {'maxErr vs cellavg':>18}")
    print("-" * 78)
    for I in [8, 16, 32, 64]:
        dz = L / I
        gt_truth = cell_avg_periodic(I, L, a)
        g_dtft, _ = bond_dtft_periodic(I, L, a)
        g_naive = bond_naive_realspace(I, L, a)
        g_nf, _ = bond_naive_fourier(I, L, a)

        for name, g in [("cell-avg(true)", gt_truth),
                        ("DTFT(eq30)", g_dtft),
                        ("naive-realspc", g_naive),
                        ("naive-fourier", g_nf)]:
            prob, rms2 = constraints(g, I, L)
            err = np.max(np.abs(g - gt_truth))
            print(f"{I:>4} {dz/a:>6.3f} | {name:<14} "
                  f"{prob:>11.6f} {rms2:>11.6f} {err:>18.3e}")
        # key comparison
        diff_dtft = np.max(np.abs(g_dtft - gt_truth))
        diff_naive = np.max(np.abs(g_naive - g_dtft))
        print(f"     {'':6} | --> DTFT matches analytic cell-avg to {diff_dtft:.2e}; "
              f"DTFT vs naive-realspace max diff = {diff_naive:.2e}")
        print("-" * 78)


# ---------------------------------------------------------------------------
# 3D version (Eq. 30 exactly, with BS replacement). Small grid.
# ---------------------------------------------------------------------------
def bond_dtft_3d_periodic(nx, lx, a, n_alias=6):
    I, J, K = nx
    Lx, Ly, Lz = lx
    dV = (Lx / I) * (Ly / J) * (Lz / K)
    qi = np.arange(I)[:, None, None]
    qj = np.arange(J)[None, :, None]
    qk = np.arange(K)[None, None, :]
    gt = np.zeros((I, J, K))
    for mx in range(-n_alias, n_alias + 1):
        ax = qi - mx * I
        for my in range(-n_alias, n_alias + 1):
            ay = qj - my * J
            for mz in range(-n_alias, n_alias + 1):
                az = qk - mz * K
                S = (ax / Lx)**2 + (ay / Ly)**2 + (az / Lz)**2
                ft = np.exp(-(np.pi**2 * a * a / 6.0) * S * 4.0 / 4.0)
                # NOTE: exponent uses xi=(q-mI)/L (DFT), FT=exp(-(2/3)pi^2 a^2 xi^2)
                ft = np.exp(-(2.0 / 3.0) * np.pi**2 * a * a * S)
                gt += ft * np.sinc(ax / I) * np.sinc(ay / J) * np.sinc(az / K)
    gt = gt / dV
    g_real = np.fft.ifftn(gt).real
    return g_real, gt

def cell_avg_3d_periodic(nx, lx, a, n_img=12):
    I, J, K = nx
    Lx, Ly, Lz = lx
    def avg_1d(N, L):
        dz = L / N
        sig = a / np.sqrt(3.0)
        n = np.arange(N)
        zc = n * dz
        g = np.zeros(N)
        for m in range(-n_img, n_img + 1):
            zp = zc + m * L
            hi = (zp + dz / 2) / (sig * np.sqrt(2))
            lo = (zp - dz / 2) / (sig * np.sqrt(2))
            g += 0.5 * (erf(hi) - erf(lo)) / dz
        return g
    gx, gy, gz = avg_1d(I, Lx), avg_1d(J, Ly), avg_1d(K, Lz)
    return gx[:, None, None] * gy[None, :, None] * gz[None, None, :]

def naive_realspace_3d(nx, lx, a, n_img=12):
    I, J, K = nx
    Lx, Ly, Lz = lx
    dV = (Lx / I) * (Ly / J) * (Lz / K)
    def samp_1d(N, L):
        dz = L / N
        n = np.arange(N)
        zc = n * dz
        g = np.zeros(N)
        for m in range(-n_img, n_img + 1):
            g += gaussian_1d(zc + m * L, a)
        return g
    g = (samp_1d(I, Lx)[:, None, None] * samp_1d(J, Ly)[None, :, None]
         * samp_1d(K, Lz)[None, None, :])
    g = g / (np.sum(g) * dV)
    return g

def constraints_3d(g, nx, lx):
    I, J, K = nx
    Lx, Ly, Lz = lx
    dV = (Lx / I) * (Ly / J) * (Lz / K)
    def coord(N, L):
        z = np.arange(N) * (L / N)
        return np.where(z > L / 2, z - L, z)
    x = coord(I, Lx)[:, None, None]
    y = coord(J, Ly)[None, :, None]
    z = coord(K, Lz)[None, None, :]
    r2 = x * x + y * y + z * z
    prob = np.sum(g) * dV
    rms2 = np.sum(r2 * g) * dV
    return prob, rms2

def run_3d():
    a = 1.0
    print()
    print("=" * 78)
    print("3D spring-bead bond function (Eq. 30, BS replacement), a=1.0")
    print("=" * 78)
    for N in [16, 24, 32]:
        nx = (N, N, N)
        lx = (8.0, 8.0, 8.0)
        gt_truth = cell_avg_3d_periodic(nx, lx, a)
        g_dtft, _ = bond_dtft_3d_periodic(nx, lx, a)
        g_naive = naive_realspace_3d(nx, lx, a)
        dz = lx[0] / N
        print(f"\n grid {N}^3, box {lx[0]}^3, dz/a = {dz/a:.3f}")
        for name, g in [("cell-avg(true)", gt_truth),
                        ("DTFT(eq30)", g_dtft),
                        ("naive-realspc", g_naive)]:
            prob, rms2 = constraints_3d(g, nx, lx)
            err = np.max(np.abs(g - gt_truth))
            print(f"   {name:<14} prob(24a)={prob:.6f}  rms^2(24b)={rms2:.6f}  "
                  f"min(g)={g.min():.3e}  maxErr_vs_true={err:.3e}")
        print(f"   --> DTFT vs naive-realspace: max|diff|={np.max(np.abs(g_dtft-g_naive)):.3e}, "
              f"L2={np.sqrt(np.sum((g_dtft-g_naive)**2)):.3e}")


def make_figure(fname="bond_dtft_comparison.png"):
    """Three-way comparison of spring-bead bond function discretizations:
       (1) Fourier-sampled  : g~(k)=exp(-(2/3)pi^2 a^2 xi^2)   [EXISTING CODE]
       (2) real-space sampled: g_n = g(z_n), normalized          [naive real space]
       (3) paper cell-average: DTFT Eq.(30), BS replacement      [Park JCP 2019]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = 1.0
    L = 8.0 * a
    C1, C2, C3 = "tab:blue", "tab:orange", "tab:green"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # ---- (a) real-space bond function at a typical grid (dz/a = 0.5) ----
    ax = axes[0, 0]
    I = 16
    dz = L / I
    z = np.arange(I) * dz
    z = np.where(z > L / 2, z - L, z)
    o = np.argsort(z)
    g_fourier, _ = bond_naive_fourier(I, L, a)
    g_real = bond_naive_realspace(I, L, a)
    g_paper, _ = bond_dtft_periodic(I, L, a)
    zc = np.linspace(-L / 2, L / 2, 400)
    ax.plot(zc, gaussian_1d(zc, a), "k-", lw=1.2, alpha=0.4, label="continuous g(z)")
    ax.plot(z[o], g_fourier[o], "o-", c=C1, label="(1) Fourier-sampled [existing]")
    ax.plot(z[o], g_real[o], "s--", c=C2, label="(2) real-space sampled")
    ax.plot(z[o], g_paper[o], "^:", c=C3, label="(3) paper cell-average (Eq.30)")
    ax.set_xlim(-3, 3)
    ax.set_xlabel("z / a"); ax.set_ylabel("g(z)")
    ax.set_title(f"Real-space bond function, dz/a = {dz/a:.2f} (I={I})")
    ax.legend(fontsize=8)

    # ---- (b) Fourier multiplier g~(k) (what the propagator actually uses) ----
    # multiplier m_q = dV * DFT(g_n);  m_0 = 1.  Plot vs k = 2*pi*q'/L.
    ax = axes[0, 1]
    q = np.arange(I)
    qp = np.where(q < I // 2, q, q - I)
    kk = 2 * np.pi * qp / L
    ks = np.argsort(kk)
    m_fourier = (dz * np.fft.fft(g_fourier)).real
    m_real = (dz * np.fft.fft(g_real)).real
    m_paper = (dz * np.fft.fft(g_paper)).real
    kcont = np.linspace(kk.min(), kk.max(), 400)
    m_cont = np.exp(-(a * a / 6.0) * kcont**2)   # exp(-b^2 k^2 ds/6), continuous
    ax.plot(kcont, m_cont, "k-", lw=1.2, alpha=0.4, label=r"continuous $e^{-a^2k^2/6}$")
    ax.plot(kk[ks], m_fourier[ks], "o-", c=C1, label="(1) Fourier-sampled [existing]")
    ax.plot(kk[ks], m_real[ks], "s--", c=C2, label="(2) real-space sampled")
    ax.plot(kk[ks], m_paper[ks], "^:", c=C3, label="(3) paper cell-average")
    ax.set_xlabel("k  (= 2π·q'/L)"); ax.set_ylabel(r"Fourier multiplier $\tilde g(k)$")
    ax.set_title("Fourier multiplier used in propagator")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- (c) rms step size (Eq. 24b, 1D target a^2/3) vs dz/a ----
    ax = axes[1, 0]
    Is = np.array([8, 12, 16, 24, 32, 48, 64, 96, 128])
    dzs = []
    rms_f, rms_r, rms_p = [], [], []
    df_fr, df_pr, df_pf = [], [], []
    for II in Is:
        ddz = L / II
        dzs.append(ddz / a)
        gf, _ = bond_naive_fourier(II, L, a)
        gr = bond_naive_realspace(II, L, a)
        gp, _ = bond_dtft_periodic(II, L, a)
        rms_f.append(constraints(gf, II, L)[1])
        rms_r.append(constraints(gr, II, L)[1])
        rms_p.append(constraints(gp, II, L)[1])
        df_fr.append(np.max(np.abs(gf - gr)))
        df_pr.append(np.max(np.abs(gp - gr)))
        df_pf.append(np.max(np.abs(gp - gf)))
    ax.semilogx(dzs, rms_f, "o-", c=C1, label="(1) Fourier-sampled")
    ax.semilogx(dzs, rms_r, "s-", c=C2, label="(2) real-space sampled")
    ax.semilogx(dzs, rms_p, "^-", c=C3, label="(3) paper cell-average")
    ax.axhline(a * a / 3.0, color="k", ls="--", lw=1, label=r"target $a^2/3$")
    ax.set_xlabel("dz / a"); ax.set_ylabel(r"$\sum z^2 g\,\Delta z$  (Eq. 24b, 1D)")
    ax.set_ylim(0.28, 0.45)
    ax.set_title("RMS step size constraint")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- (d) pairwise max difference between the three methods ----
    ax = axes[1, 1]
    ax.loglog(dzs, df_fr, "o-", c="purple", label="|Fourier - real-space|")
    ax.loglog(dzs, df_pr, "s-", c=C3, label="|paper - real-space|")
    ax.loglog(dzs, df_pf, "^-", c="tab:red", label="|paper - Fourier|")
    ax.set_xlabel("dz / a"); ax.set_ylabel("max |Δg|")
    ax.set_title("Pairwise difference between the 3 methods")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Spring-bead bond function — 3 discretizations: "
                 "(1) Fourier-sampled [existing]  (2) real-space  (3) paper cell-average",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=130)
    print(f"\nFigure saved to {fname}")


def make_2d_slice_figure(I=160, L=8.0, a=1.0, half_window=2.0,
                         fname="bond_2d_slice.png"):
    """2D cross-section (z=0 plane) of the 3D spring-bead bond function for the
    three methods, plus difference maps. The Gaussian bond function is separable,
    so g_3D[i,j,k] = g_1D[i]*g_1D[j]*g_1D[k]; the z=0 slice = outer(g1d,g1d)*g1d[0].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dz = L / I
    # 1D bond functions for the 3 methods
    gf, _ = bond_naive_fourier(I, L, a)
    gr = bond_naive_realspace(I, L, a)
    gp, _ = bond_dtft_periodic(I, L, a)

    z = np.arange(I) * dz
    z = np.where(z > L / 2, z - L, z)
    sh = np.argsort(z)               # shift so center is in the middle
    zc = z[sh]
    sel = np.abs(zc) <= half_window
    zc = zc[sel]

    def slice2d(g1d):
        g = g1d[sh][sel]
        return np.outer(g, g) * g1d[0]   # z=0 plane

    Sf, Sr, Sp = slice2d(gf), slice2d(gr), slice2d(gp)
    ext = [zc.min(), zc.max(), zc.min(), zc.max()]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

    vmax = max(Sf.max(), Sr.max(), Sp.max())
    for ax, S, t in [(axes[0, 0], Sf, "(1) Fourier-sampled [existing]"),
                     (axes[0, 1], Sr, "(2) real-space sampled"),
                     (axes[0, 2], Sp, "(3) paper cell-average (Eq.30)")]:
        im = ax.imshow(S, extent=ext, origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_title(t); ax.set_xlabel("x / a"); ax.set_ylabel("y / a")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # difference maps (row 2)
    dpr = Sp - Sr      # paper - real-space (the meaningful difference)
    dfr = Sf - Sr      # Fourier - real-space (machine zero)
    m = np.max(np.abs(dpr))
    for ax, D, t in [(axes[1, 0], dpr, "Δ = (3) paper − (2) real-space"),
                     (axes[1, 1], dfr, "Δ = (1) Fourier − (2) real-space")]:
        im = ax.imshow(D, extent=ext, origin="lower", cmap="RdBu_r",
                       vmin=-m, vmax=m)
        ax.set_title(f"{t}\nmax|Δ|={np.max(np.abs(D)):.2e}")
        ax.set_xlabel("x / a"); ax.set_ylabel("y / a")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # diagonal line cut through the 2D slice
    ax = axes[1, 2]
    diag = np.arange(len(zc))
    r_diag = zc * np.sqrt(2.0)
    ax.plot(r_diag, np.diag(Sf), "o-", ms=3, label="(1) Fourier")
    ax.plot(r_diag, np.diag(Sr), "s--", ms=3, label="(2) real-space")
    ax.plot(r_diag, np.diag(Sp), "^:", ms=3, label="(3) paper")
    ax.set_xlabel("diagonal distance / a"); ax.set_ylabel("g(x,y,0)")
    ax.set_title("Diagonal cut through slice")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Spring-bead bond function — z=0 cross-section, "
                 f"dz/a = {dz/a:.3f} (I={I}, L={L:.0f}a)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=130)
    print(f"\n2D slice figure saved to {fname}")


def plot_for_ds(ds=0.01, b=1.0, fname=None):
    """Bond function for a physical step size ds (a = b*sqrt(ds)).
    The discretization error depends only on dx/a, so this re-expresses the
    universal curves at a = b*sqrt(ds), and marks the typical discrete-SCFT
    grid regime (box ~ few R0 = few * b*sqrt(N), N = 1/ds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = b * np.sqrt(ds)            # discrete bond length, a^2 = b^2 ds
    N = int(round(1.0 / ds))
    R0 = b * np.sqrt(N)            # length unit = b N^{1/2} = a N^{1/2} (a=b here)
    if fname is None:
        fname = f"bond_ds_{N}.png"
    L = 20.0 * a                   # box covers the bond Gaussian (width ~a)

    # x-axis is dx in units of R0 = b N^{1/2};  dx/R0 = (dx/a) * (a/R0) = (dx/a)*ds
    to_R0 = a / R0                 # = ds

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    C1, C2, C3 = "tab:blue", "tab:orange", "tab:green"

    # ---- (a) 1D bond function at a well-resolved grid (dx/a = 0.5) ----
    ax = axes[0, 0]
    I = 40
    dz = L / I
    z = np.arange(I) * dz
    z = np.where(z > L / 2, z - L, z)
    o = np.argsort(z)
    gf, _ = bond_naive_fourier(I, L, a)
    gr = bond_naive_realspace(I, L, a)
    gp, _ = bond_dtft_periodic(I, L, a)
    zc = np.linspace(-L / 2, L / 2, 400)
    ax.plot(zc / a, gaussian_1d(zc, a) * a, "k-", lw=1, alpha=0.4, label="continuous")
    ax.plot(z[o] / a, gf[o] * a, "o-", c=C1, label="(1) Fourier [code]")
    ax.plot(z[o] / a, gr[o] * a, "s--", c=C2, label="(2) real-space")
    ax.plot(z[o] / a, gp[o] * a, "^:", c=C3, label="(3) paper")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("z / a"); ax.set_ylabel("g(z)·a (scaled)")
    ax.set_title(f"Bond fn, dx = {dz/R0:.4f} R0 (dx/a={dz/a:.2f})  ds=1/{N}")
    ax.legend(fontsize=8)

    # ---- scan over dx ----
    Is = np.array([8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 120, 160, 240])
    dxR0, minf, minr, minp = [], [], [], []
    rms_f, rms_r, rms_p = [], [], []
    for II in Is:
        ddz = L / II
        dxR0.append(ddz / R0)
        gf2, _ = bond_naive_fourier(II, L, a)
        gr2 = bond_naive_realspace(II, L, a)
        gp2, _ = bond_dtft_periodic(II, L, a)
        minf.append(gf2.min()); minr.append(gr2.min()); minp.append(gp2.min())
        rms_f.append(constraints(gf2, II, L)[1])
        rms_r.append(constraints(gr2, II, L)[1])
        rms_p.append(constraints(gp2, II, L)[1])
    dxR0 = np.array(dxR0)

    # realistic discrete-SCFT grid regime in R0 units: box ~ 3-4 R0, nx ~ 32-128
    dxR0_lo = 3.0 / 128.0
    dxR0_hi = 4.0 / 32.0

    xlab = r"$dx / (a N^{1/2}) = dx / R_0$"

    # ---- (b) minimum value of g (negativity) vs dx/R0 ----
    ax = axes[0, 1]
    ax.semilogx(dxR0, minf, "o-", c=C1, label="(1) Fourier [code]")
    ax.semilogx(dxR0, minr, "s-", c=C2, label="(2) real-space")
    ax.semilogx(dxR0, minp, "^-", c=C3, label="(3) paper")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(dxR0_lo, dxR0_hi, color="gray", alpha=0.2,
               label=f"typical SCFT grid\n(dx/R0≈{dxR0_lo:.02f}-{dxR0_hi:.02f})")
    ax.set_xlabel(xlab); ax.set_ylabel("min g(z)")
    ax.set_title("Negativity of bond function")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- (c) rms step (Eq. 24b, target a^2/3) vs dx/R0 ----
    ax = axes[1, 0]
    ax.semilogx(dxR0, np.array(rms_f) / (a * a), "o-", c=C1, label="(1) Fourier")
    ax.semilogx(dxR0, np.array(rms_r) / (a * a), "s-", c=C2, label="(2) real-space")
    ax.semilogx(dxR0, np.array(rms_p) / (a * a), "^-", c=C3, label="(3) paper")
    ax.axhline(1.0 / 3.0, color="k", ls="--", lw=1, label="target 1/3")
    ax.axvspan(dxR0_lo, dxR0_hi, color="gray", alpha=0.2)
    ax.set_xlabel(xlab); ax.set_ylabel(r"$\langle z^2\rangle / a^2$")
    ax.set_title("RMS step size (Eq. 24b)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- (d) 1D bond function at a coarse/realistic grid ----
    ax = axes[1, 1]
    dx_target = 0.5 * (dxR0_lo + dxR0_hi) * R0
    Ic = max(6, int(round(L / dx_target)))
    dzc = L / Ic
    zc2 = np.arange(Ic) * dzc
    zc2 = np.where(zc2 > L / 2, zc2 - L, zc2)
    oc = np.argsort(zc2)
    gfc, _ = bond_naive_fourier(Ic, L, a)
    grc = bond_naive_realspace(Ic, L, a)
    gpc, _ = bond_dtft_periodic(Ic, L, a)
    ax.plot(zc2[oc] / a, gfc[oc] * a, "o-", c=C1, label="(1) Fourier [code]")
    ax.plot(zc2[oc] / a, grc[oc] * a, "s--", c=C2, label="(2) real-space")
    ax.plot(zc2[oc] / a, gpc[oc] * a, "^:", c=C3, label="(3) paper")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("z / a"); ax.set_ylabel("g(z)·a (scaled)")
    ax.set_title(f"Coarse/realistic grid, dx = {dzc/R0:.3f} R0 (dx/a={dzc/a:.1f})")
    ax.legend(fontsize=8)

    fig.suptitle(f"Spring-bead bond function at ds = 1/{N}  (a = {a:.3g}b, "
                 f"R0 = a N$^{{1/2}}$ = {R0:.1f}b);  x-axis in R0 units;  "
                 f"code uses method (1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=130)
    print(f"\nFigure saved to {fname}")
    print(f"ds=1/{N}: a={a:.4g}b, R0={R0:.3g}b, typical SCFT dx/R0 ~ "
          f"{dxR0_lo:.3f}-{dxR0_hi:.3f}")
    return fig


if __name__ == "__main__":
    run_1d()
    run_3d()
    make_figure()
    make_2d_slice_figure()
    plot_for_ds(ds=0.01)
