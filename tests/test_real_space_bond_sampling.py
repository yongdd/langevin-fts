#!/usr/bin/env python3
"""
Demonstrate real-space sampling of the bond function.

The Boltzmann bond factor in Fourier space is:
    g̃(k) = exp(-b²|k|²ds/6)

Its inverse Fourier transform (the real-space bond function) is:
    g(r) = (3/(2πb²ds))^(d/2) exp(-3|r|²/(2b²ds))

This script:
1. Samples the Gaussian in real space
2. Applies DFT to get discretized Fourier coefficients
3. Compares with the analytical exp(-b²|k|²ds/6)
4. Verifies that IDFT of sampled coefficients is non-negative
"""

import numpy as np
import matplotlib.pyplot as plt

def real_space_bond_1d(r, b, ds):
    """Real-space bond function in 1D."""
    a2 = b * b * ds  # a² = b²ds
    return np.sqrt(3.0 / (2.0 * np.pi * a2)) * np.exp(-3.0 * r**2 / (2.0 * a2))

def real_space_bond_2d(rx, ry, b, ds):
    """Real-space bond function in 2D."""
    a2 = b * b * ds
    r2 = rx**2 + ry**2
    return (3.0 / (2.0 * np.pi * a2)) * np.exp(-3.0 * r2 / (2.0 * a2))

def real_space_bond_3d(rx, ry, rz, b, ds):
    """Real-space bond function in 3D."""
    a2 = b * b * ds
    r2 = rx**2 + ry**2 + rz**2
    return (3.0 / (2.0 * np.pi * a2))**(3.0/2.0) * np.exp(-3.0 * r2 / (2.0 * a2))

def analytical_boltz_bond_1d(k, b, ds):
    """Analytical Boltzmann bond factor in 1D."""
    return np.exp(-b**2 * k**2 * ds / 6.0)

def test_1d_sampling():
    """Test 1D real-space sampling approach."""
    print("=" * 60)
    print("1D Test: Real-space sampling of bond function")
    print("=" * 60)

    # Parameters
    N = 64
    L = 4.0
    b = 1.0
    ds = 0.01

    dx = L / N

    # Real-space grid (cell-centered)
    x = (np.arange(N) + 0.5) * dx

    # For periodic BC, use minimum image convention
    # Distance from origin, wrapped at L/2
    r = x.copy()
    r[r > L/2] -= L

    # Sample the Gaussian in real space
    g_real = real_space_bond_1d(r, b, ds)

    # Apply DFT (with proper normalization)
    g_fourier = np.fft.fft(g_real) * dx  # multiply by dx for continuous FT convention

    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, dx)

    # Analytical Boltzmann bond factor
    g_analytical = analytical_boltz_bond_1d(k, b, ds)

    # Inverse DFT to verify non-negativity
    g_back = np.fft.ifft(g_fourier / dx).real

    print(f"Grid: N={N}, L={L}, dx={dx:.4f}")
    print(f"Bond parameters: b={b}, ds={ds}")
    print(f"Gaussian width: sigma = sqrt(b²ds/3) = {np.sqrt(b**2*ds/3):.4f}")
    print(f"sigma/dx = {np.sqrt(b**2*ds/3)/dx:.4f}")
    print()
    print(f"Real-space samples: min={g_real.min():.6e}, max={g_real.max():.6e}")
    print(f"IDFT result: min={g_back.min():.6e}, max={g_back.max():.6e}")
    print(f"All IDFT values non-negative: {np.all(g_back >= -1e-15)}")
    print()

    # Compare Fourier coefficients
    # Only k=0 component should be exactly 1 for normalized bond function
    print(f"k=0 component (should be ~1): {g_fourier[0].real:.6f}")
    print(f"Comparison at k=0: analytical={g_analytical[0]:.6f}, sampled={g_fourier[0].real:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Real-space bond function
    ax = axes[0, 0]
    r_fine = np.linspace(-L/2, L/2, 1000)
    ax.plot(r_fine, real_space_bond_1d(r_fine, b, ds), 'b-', label='Analytical')
    ax.plot(r, g_real, 'ro', markersize=4, label='Sampled')
    ax.set_xlabel('r')
    ax.set_ylabel('g(r)')
    ax.set_title('Real-space bond function')
    ax.legend()
    ax.grid(True)

    # Fourier coefficients comparison
    ax = axes[0, 1]
    k_sorted_idx = np.argsort(k)
    ax.plot(k[k_sorted_idx], g_analytical[k_sorted_idx], 'b-', linewidth=2, label='Analytical exp(-b²k²ds/6)')
    ax.plot(k[k_sorted_idx], g_fourier[k_sorted_idx].real, 'r--', linewidth=2, label='DFT of sampled g(r)')
    ax.set_xlabel('k')
    ax.set_ylabel('g̃(k)')
    ax.set_title('Fourier-space bond function')
    ax.legend()
    ax.grid(True)

    # Difference in Fourier space
    ax = axes[1, 0]
    diff = g_fourier.real - g_analytical
    ax.plot(k[k_sorted_idx], diff[k_sorted_idx], 'g-', linewidth=2)
    ax.set_xlabel('k')
    ax.set_ylabel('g̃_sampled - g̃_analytical')
    ax.set_title('Difference in Fourier coefficients')
    ax.grid(True)

    # IDFT verification
    ax = axes[1, 1]
    ax.plot(x, g_back, 'b-', linewidth=2, label='IDFT of sampled coefficients')
    ax.axhline(y=0, color='r', linestyle='--', label='Zero line')
    ax.set_xlabel('x')
    ax.set_ylabel('g(x)')
    ax.set_title(f'IDFT result (min={g_back.min():.2e})')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('test_real_space_bond_1d.png', dpi=150)
    print(f"\nPlot saved to test_real_space_bond_1d.png")

    return g_real, g_fourier, g_analytical

def test_1d_varying_resolution():
    """Test how sampled bond function behaves with different resolutions."""
    print("\n" + "=" * 60)
    print("1D Resolution Study")
    print("=" * 60)

    L = 4.0
    b = 1.0
    ds = 0.01

    resolutions = [16, 32, 64, 128, 256]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, N in enumerate(resolutions):
        dx = L / N
        sigma = np.sqrt(b**2 * ds / 3)

        # Real-space grid
        x = (np.arange(N) + 0.5) * dx
        r = x.copy()
        r[r > L/2] -= L

        # Sample Gaussian
        g_real = real_space_bond_1d(r, b, ds)

        # DFT
        g_fourier = np.fft.fft(g_real) * dx

        # Wavenumbers
        k = 2 * np.pi * np.fft.fftfreq(N, dx)

        # Analytical
        g_analytical = analytical_boltz_bond_1d(k, b, ds)

        # IDFT
        g_back = np.fft.ifft(g_fourier / dx).real

        print(f"N={N:4d}, dx={dx:.4f}, sigma/dx={sigma/dx:.2f}, "
              f"IDFT min={g_back.min():+.2e}, "
              f"max |diff|={np.max(np.abs(g_fourier.real - g_analytical)):.2e}")

        # Plot
        if idx < 5:
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            k_sorted = np.argsort(k)
            ax.plot(k[k_sorted], g_analytical[k_sorted], 'b-', label='Analytical')
            ax.plot(k[k_sorted], g_fourier[k_sorted].real, 'r--', label='Sampled')
            ax.set_title(f'N={N}, σ/dx={sigma/dx:.2f}')
            ax.set_xlabel('k')
            ax.set_ylabel('g̃(k)')
            ax.legend()
            ax.grid(True)

    # Last plot: show difference at high k
    ax = axes[1, 2]
    for N in resolutions:
        dx = L / N
        x = (np.arange(N) + 0.5) * dx
        r = x.copy()
        r[r > L/2] -= L
        g_real = real_space_bond_1d(r, b, ds)
        g_fourier = np.fft.fft(g_real) * dx
        k = 2 * np.pi * np.fft.fftfreq(N, dx)
        g_analytical = analytical_boltz_bond_1d(k, b, ds)
        k_sorted = np.argsort(k)
        diff = np.abs(g_fourier.real - g_analytical)
        ax.semilogy(k[k_sorted], diff[k_sorted], label=f'N={N}')
    ax.set_xlabel('k')
    ax.set_ylabel('|g̃_sampled - g̃_analytical|')
    ax.set_title('Difference vs resolution')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('test_real_space_bond_resolution.png', dpi=150)
    print(f"\nPlot saved to test_real_space_bond_resolution.png")

def test_2d_sampling():
    """Test 2D real-space sampling."""
    print("\n" + "=" * 60)
    print("2D Test: Real-space sampling of bond function")
    print("=" * 60)

    # Parameters
    Nx, Ny = 32, 32
    Lx, Ly = 4.0, 4.0
    b = 1.0
    ds = 0.01

    dx, dy = Lx / Nx, Ly / Ny

    # Real-space grid (cell-centered)
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    # Minimum image
    rx = x.copy()
    rx[rx > Lx/2] -= Lx
    ry = y.copy()
    ry[ry > Ly/2] -= Ly

    RX, RY = np.meshgrid(rx, ry, indexing='ij')

    # Sample Gaussian
    g_real = real_space_bond_2d(RX, RY, b, ds)

    # 2D DFT
    g_fourier = np.fft.fft2(g_real) * dx * dy

    # Wavenumbers
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    # Analytical
    g_analytical = np.exp(-b**2 * K2 * ds / 6.0)

    # IDFT
    g_back = np.fft.ifft2(g_fourier / (dx * dy)).real

    print(f"Grid: {Nx}x{Ny}, Lx={Lx}, Ly={Ly}")
    print(f"Real-space: min={g_real.min():.6e}, max={g_real.max():.6e}")
    print(f"IDFT: min={g_back.min():.6e}, max={g_back.max():.6e}")
    print(f"All IDFT values non-negative: {np.all(g_back >= -1e-15)}")
    print(f"Max |diff| in Fourier space: {np.max(np.abs(g_fourier.real - g_analytical)):.6e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    im = ax.imshow(g_real.T, origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar(im, ax=ax)
    ax.set_title('Real-space g(r) (sampled)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = axes[0, 1]
    im = ax.imshow(np.fft.fftshift(g_fourier.real).T, origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_title('Fourier coefficients (DFT of samples)')

    ax = axes[1, 0]
    im = ax.imshow(np.fft.fftshift(g_analytical).T, origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_title('Analytical exp(-b²|k|²ds/6)')

    ax = axes[1, 1]
    diff = np.abs(g_fourier.real - g_analytical)
    im = ax.imshow(np.fft.fftshift(diff).T, origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_title('|Difference|')

    plt.tight_layout()
    plt.savefig('test_real_space_bond_2d.png', dpi=150)
    print(f"\nPlot saved to test_real_space_bond_2d.png")

def test_normalization():
    """Test that sampled bond function preserves normalization."""
    print("\n" + "=" * 60)
    print("Normalization Test")
    print("=" * 60)

    # 1D test
    N = 64
    L = 4.0
    b = 1.0
    ds = 0.01
    dx = L / N

    x = (np.arange(N) + 0.5) * dx
    r = x.copy()
    r[r > L/2] -= L

    g_real = real_space_bond_1d(r, b, ds)

    # Numerical integral (should be ~1)
    integral = np.sum(g_real) * dx
    print(f"1D: ∫g(r)dr = {integral:.6f} (should be 1)")

    # k=0 component of DFT (should also be ~1)
    g_fourier = np.fft.fft(g_real) * dx
    print(f"1D: g̃(k=0) = {g_fourier[0].real:.6f} (should be 1)")

    # Second moment
    second_moment = np.sum(r**2 * g_real) * dx
    expected_moment = b**2 * ds / 3  # <r²> = a²/3 for 1D Gaussian with variance a²/3
    print(f"1D: <r²> = {second_moment:.6f}, expected = {expected_moment:.6f}")

if __name__ == "__main__":
    test_1d_sampling()
    test_1d_varying_resolution()
    test_2d_sampling()
    test_normalization()

    plt.show()
