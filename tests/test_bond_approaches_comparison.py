#!/usr/bin/env python3
"""
Compare different approaches to discretizing the bond function:

1. Analytical: exp(-b²|k|²ds/6) - the continuous formula
2. Real-space sampling: DFT of sampled Gaussian in real space
3. Sinc filtering: exp(-b²|k|²ds/6) × sinc(kx·dx) × sinc(ky·dy) × ...
   (cell-averaging approach from Park et al. 2019)

The goal is to see which approach gives better accuracy for propagator computation.
"""

import numpy as np
import matplotlib.pyplot as plt

def sinc_unnormalized(x):
    """Unnormalized sinc: sin(x)/x with sinc(0)=1."""
    return np.sinc(x / np.pi)

def test_1d_propagator_accuracy():
    """
    Test propagator accuracy with different bond function approaches.

    We solve: dq/ds = (b²/6)∇²q - w·q
    with w = 0 and initial condition q(x,0) = cos(2πx/L)

    Analytical solution: q(x,s) = cos(2πx/L) · exp(-b²(2π/L)²s/6)
    """
    print("=" * 70)
    print("1D Propagator Accuracy Test")
    print("=" * 70)

    # Parameters
    N = 32
    L = 4.0
    b = 1.0
    ds = 0.1  # Large ds to see differences
    n_steps = 10

    dx = L / N

    # Real-space grid (cell-centered)
    x = (np.arange(N) + 0.5) * dx

    # Initial condition
    k0 = 2 * np.pi / L
    q0 = np.cos(k0 * x)

    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, dx)

    # ========================================
    # Approach 1: Analytical bond function
    # ========================================
    boltz_analytical = np.exp(-b**2 * k**2 * ds / 6.0)

    q_analytical = q0.copy()
    for _ in range(n_steps):
        qk = np.fft.fft(q_analytical)
        qk *= boltz_analytical
        q_analytical = np.fft.ifft(qk).real

    # ========================================
    # Approach 2: Real-space sampled bond function
    # ========================================
    # Sample Gaussian in real space
    r = x.copy()
    r[r > L/2] -= L
    sigma2 = b**2 * ds / 3  # variance
    g_real = np.sqrt(3.0 / (2.0 * np.pi * b**2 * ds)) * np.exp(-3.0 * r**2 / (2.0 * b**2 * ds))

    # DFT to get bond function in Fourier space
    boltz_sampled = np.fft.fft(g_real) * dx

    q_sampled = q0.copy()
    for _ in range(n_steps):
        qk = np.fft.fft(q_sampled)
        qk *= boltz_sampled
        q_sampled = np.fft.ifft(qk).real

    # ========================================
    # Approach 3: Sinc-filtered bond function
    # ========================================
    boltz_sinc = np.exp(-b**2 * k**2 * ds / 6.0) * sinc_unnormalized(k * dx / 2)

    q_sinc = q0.copy()
    for _ in range(n_steps):
        qk = np.fft.fft(q_sinc)
        qk *= boltz_sinc
        q_sinc = np.fft.ifft(qk).real

    # ========================================
    # Exact solution
    # ========================================
    s_total = n_steps * ds
    q_exact = np.cos(k0 * x) * np.exp(-b**2 * k0**2 * s_total / 6.0)

    # ========================================
    # Compare
    # ========================================
    print(f"Parameters: N={N}, L={L}, dx={dx:.4f}, ds={ds}, n_steps={n_steps}")
    print(f"Gaussian width: sigma={np.sqrt(sigma2):.4f}, sigma/dx={np.sqrt(sigma2)/dx:.2f}")
    print()

    err_analytical = np.max(np.abs(q_analytical - q_exact))
    err_sampled = np.max(np.abs(q_sampled - q_exact))
    err_sinc = np.max(np.abs(q_sinc - q_exact))

    print(f"Max error (analytical bond):      {err_analytical:.6e}")
    print(f"Max error (real-space sampled):   {err_sampled:.6e}")
    print(f"Max error (sinc-filtered):        {err_sinc:.6e}")

    # Check if IDFT of sampled coefficients is non-negative
    g_back = np.fft.ifft(boltz_sampled / dx).real
    print(f"\nIDFT of sampled bond: min={g_back.min():.2e}, max={g_back.max():.2e}")

    g_back_sinc = np.fft.ifft(boltz_sinc * N).real  # N = 1/dx for discrete
    print(f"IDFT of sinc-filtered: min={g_back_sinc.min():.2e}, max={g_back_sinc.max():.2e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(x, q_exact, 'k-', linewidth=2, label='Exact')
    ax.plot(x, q_analytical, 'b--', linewidth=2, label='Analytical bond')
    ax.plot(x, q_sampled, 'r:', linewidth=2, label='Real-space sampled')
    ax.plot(x, q_sinc, 'g-.', linewidth=2, label='Sinc-filtered')
    ax.set_xlabel('x')
    ax.set_ylabel('q(x,s)')
    ax.set_title(f'Propagator after {n_steps} steps (s={s_total})')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(x, q_analytical - q_exact, 'b-', label='Analytical')
    ax.plot(x, q_sampled - q_exact, 'r-', label='Sampled')
    ax.plot(x, q_sinc - q_exact, 'g-', label='Sinc')
    ax.set_xlabel('x')
    ax.set_ylabel('Error')
    ax.set_title('Error in propagator')
    ax.legend()
    ax.grid(True)

    # Bond functions in Fourier space
    ax = axes[1, 0]
    k_sorted = np.argsort(k)
    ax.plot(k[k_sorted], boltz_analytical[k_sorted], 'b-', linewidth=2, label='Analytical')
    ax.plot(k[k_sorted], boltz_sampled[k_sorted].real, 'r--', linewidth=2, label='Sampled')
    ax.plot(k[k_sorted], boltz_sinc[k_sorted], 'g:', linewidth=2, label='Sinc-filtered')
    ax.set_xlabel('k')
    ax.set_ylabel('g̃(k)')
    ax.set_title('Bond function in Fourier space')
    ax.legend()
    ax.grid(True)

    # Real-space bond functions
    ax = axes[1, 1]
    ax.plot(x, g_back, 'r-', linewidth=2, label='IDFT of sampled')
    ax.plot(x, g_back_sinc, 'g-', linewidth=2, label='IDFT of sinc-filtered')
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set_xlabel('x')
    ax.set_ylabel('g(x)')
    ax.set_title('Real-space bond function (IDFT)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('test_bond_approaches_comparison.png', dpi=150)
    print(f"\nPlot saved to test_bond_approaches_comparison.png")

    return err_analytical, err_sampled, err_sinc

def test_convergence_with_resolution():
    """Test how errors scale with resolution."""
    print("\n" + "=" * 70)
    print("Convergence with Resolution")
    print("=" * 70)

    L = 4.0
    b = 1.0
    ds = 0.01
    n_steps = 100

    resolutions = [16, 32, 64, 128, 256, 512]

    errors_analytical = []
    errors_sampled = []
    errors_sinc = []

    for N in resolutions:
        dx = L / N
        x = (np.arange(N) + 0.5) * dx

        # Initial condition
        k0 = 2 * np.pi / L
        q0 = np.cos(k0 * x)

        # Wavenumbers
        k = 2 * np.pi * np.fft.fftfreq(N, dx)

        # Bond functions
        boltz_analytical = np.exp(-b**2 * k**2 * ds / 6.0)

        r = x.copy()
        r[r > L/2] -= L
        g_real = np.sqrt(3.0 / (2.0 * np.pi * b**2 * ds)) * np.exp(-3.0 * r**2 / (2.0 * b**2 * ds))
        boltz_sampled = np.fft.fft(g_real) * dx

        boltz_sinc = np.exp(-b**2 * k**2 * ds / 6.0) * sinc_unnormalized(k * dx / 2)

        # Propagate
        q_a = q0.copy()
        q_s = q0.copy()
        q_c = q0.copy()

        for _ in range(n_steps):
            qk = np.fft.fft(q_a)
            qk *= boltz_analytical
            q_a = np.fft.ifft(qk).real

            qk = np.fft.fft(q_s)
            qk *= boltz_sampled
            q_s = np.fft.ifft(qk).real

            qk = np.fft.fft(q_c)
            qk *= boltz_sinc
            q_c = np.fft.ifft(qk).real

        # Exact
        s_total = n_steps * ds
        q_exact = np.cos(k0 * x) * np.exp(-b**2 * k0**2 * s_total / 6.0)

        errors_analytical.append(np.max(np.abs(q_a - q_exact)))
        errors_sampled.append(np.max(np.abs(q_s - q_exact)))
        errors_sinc.append(np.max(np.abs(q_c - q_exact)))

        sigma = np.sqrt(b**2 * ds / 3)
        print(f"N={N:4d}, dx={dx:.4f}, σ/dx={sigma/dx:.2f}: "
              f"err_ana={errors_analytical[-1]:.2e}, "
              f"err_samp={errors_sampled[-1]:.2e}, "
              f"err_sinc={errors_sinc[-1]:.2e}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(resolutions, errors_analytical, 'bo-', label='Analytical')
    plt.loglog(resolutions, errors_sampled, 'rs-', label='Real-space sampled')
    plt.loglog(resolutions, errors_sinc, 'g^-', label='Sinc-filtered')
    plt.xlabel('N (grid points)')
    plt.ylabel('Max error')
    plt.title(f'Convergence with resolution (ds={ds}, n_steps={n_steps})')
    plt.legend()
    plt.grid(True)
    plt.savefig('test_bond_convergence.png', dpi=150)
    print(f"\nPlot saved to test_bond_convergence.png")

def test_strong_field():
    """Test with a strong potential field w(r)."""
    print("\n" + "=" * 70)
    print("Test with Strong Potential Field")
    print("=" * 70)

    N = 64
    L = 4.0
    b = 1.0
    ds = 0.01
    n_steps = 100

    dx = L / N
    x = (np.arange(N) + 0.5) * dx

    # Strong random field
    np.random.seed(42)
    w = np.random.normal(0, 5.0, N)

    # Initial condition
    q0 = np.ones(N)

    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, dx)

    # Bond functions
    boltz_analytical = np.exp(-b**2 * k**2 * ds / 6.0)

    r = x.copy()
    r[r > L/2] -= L
    g_real = np.sqrt(3.0 / (2.0 * np.pi * b**2 * ds)) * np.exp(-3.0 * r**2 / (2.0 * b**2 * ds))
    boltz_sampled = np.fft.fft(g_real) * dx

    boltz_sinc = np.exp(-b**2 * k**2 * ds / 6.0) * sinc_unnormalized(k * dx / 2)

    # Propagate with operator splitting
    exp_w_half = np.exp(-w * ds / 2)

    q_a = q0.copy()
    q_s = q0.copy()
    q_c = q0.copy()

    for step in range(n_steps):
        # Analytical
        q_a *= exp_w_half
        qk = np.fft.fft(q_a)
        qk *= boltz_analytical
        q_a = np.fft.ifft(qk).real
        q_a *= exp_w_half

        # Sampled
        q_s *= exp_w_half
        qk = np.fft.fft(q_s)
        qk *= boltz_sampled
        q_s = np.fft.ifft(qk).real
        q_s *= exp_w_half

        # Sinc
        q_c *= exp_w_half
        qk = np.fft.fft(q_c)
        qk *= boltz_sinc
        q_c = np.fft.ifft(qk).real
        q_c *= exp_w_half

    print(f"After {n_steps} steps with strong field (std(w)=5):")
    print(f"  Analytical: min={q_a.min():.6e}, max={q_a.max():.6e}")
    print(f"  Sampled:    min={q_s.min():.6e}, max={q_s.max():.6e}")
    print(f"  Sinc:       min={q_c.min():.6e}, max={q_c.max():.6e}")

    # Check for negative values
    print(f"\n  Analytical has negative values: {np.any(q_a < 0)}")
    print(f"  Sampled has negative values:    {np.any(q_s < 0)}")
    print(f"  Sinc has negative values:       {np.any(q_c < 0)}")

    if np.any(q_a < 0):
        print(f"    Most negative (analytical): {q_a.min():.6e}")
    if np.any(q_s < 0):
        print(f"    Most negative (sampled): {q_s.min():.6e}")
    if np.any(q_c < 0):
        print(f"    Most negative (sinc): {q_c.min():.6e}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(x, w, 'k-')
    ax.set_xlabel('x')
    ax.set_ylabel('w(x)')
    ax.set_title('Potential field')
    ax.grid(True)

    ax = axes[1]
    ax.plot(x, q_a, 'b-', label='Analytical')
    ax.plot(x, q_s, 'r--', label='Sampled')
    ax.plot(x, q_c, 'g:', label='Sinc')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('q(x)')
    ax.set_title('Propagator')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ax.plot(x, q_s - q_a, 'r-', label='Sampled - Analytical')
    ax.plot(x, q_c - q_a, 'g-', label='Sinc - Analytical')
    ax.set_xlabel('x')
    ax.set_ylabel('Difference')
    ax.set_title('Difference from analytical')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('test_strong_field.png', dpi=150)
    print(f"\nPlot saved to test_strong_field.png")

if __name__ == "__main__":
    test_1d_propagator_accuracy()
    test_convergence_with_resolution()
    test_strong_field()
    plt.show()
