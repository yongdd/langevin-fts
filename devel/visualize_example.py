"""
Example: Visualizing SCFT simulation results.

This script demonstrates how to visualize polymer density fields
after running an SCFT simulation.

For more visualization tools, see the `tools/` directory:
- plot_2d_density.ipynb: 2D density plots with matplotlib
- plot_3d_isodensity.ipynb: 3D isosurface with plotly

Requirements:
    pip install matplotlib
    # Optional for 3D:
    pip install plotly
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless systems
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from polymerfts import scft

# Suppress OpenMP warnings
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"


def visualize_2d(phi_dict, lx, filename=None):
    """Visualize 2D concentration fields.

    Based on tools/plot_2d_density.ipynb
    """
    n_fields = len(phi_dict)
    fig, axes = plt.subplots(1, n_fields, figsize=(4*n_fields, 4))
    fig.suptitle("Concentration", fontsize=16)

    if n_fields == 1:
        axes = [axes]

    for ax, (name, phi) in zip(axes, phi_dict.items()):
        im = ax.imshow(phi.T, extent=(0, lx[0], 0, lx[1]),
                       origin='lower', cmap=cm.jet, vmin=0.0, vmax=1.0)
        ax.set(title=f'φ_{name}', xlabel='x', ylabel='y')

    fig.subplots_adjust(right=0.85)
    fig.colorbar(im, ax=axes, shrink=0.8)

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    else:
        plt.show()
    plt.close()


def visualize_3d_slices(phi, lx, title="Concentration", filename=None):
    """Visualize 3D field as orthogonal slices."""
    nx, ny, nz = phi.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # XY slice (z = middle)
    axes[0].imshow(phi[:, :, nz//2].T, origin='lower',
                   extent=[0, lx[0], 0, lx[1]], cmap=cm.jet, vmin=0.0, vmax=1.0)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'{title} (z = {lx[2]/2:.2f})')

    # XZ slice (y = middle)
    axes[1].imshow(phi[:, ny//2, :].T, origin='lower',
                   extent=[0, lx[0], 0, lx[2]], cmap=cm.jet, vmin=0.0, vmax=1.0)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    axes[1].set_title(f'{title} (y = {lx[1]/2:.2f})')

    # YZ slice (x = middle)
    im = axes[2].imshow(phi[nx//2, :, :].T, origin='lower',
                        extent=[0, lx[1], 0, lx[2]], cmap=cm.jet, vmin=0.0, vmax=1.0)
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')
    axes[2].set_title(f'{title} (x = {lx[0]/2:.2f})')

    fig.colorbar(im, ax=axes, label='φ', shrink=0.8)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    else:
        plt.show()
    plt.close()


def visualize_3d_isosurface(phi, nx, filename=None):
    """Visualize 3D field as isosurface using Plotly.

    Based on tools/plot_3d_isodensity.ipynb
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return

    X, Y, Z = np.mgrid[0:nx[0], 0:nx[1], 0:nx[2]]
    values = phi.reshape(nx)

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=0.0,
        isomax=1.0,
        opacity=0.2,
        surface_count=9,
        colorscale='RdBu',
        colorbar=dict(thickness=20, tickfont=dict(size=14)),
        reversescale=True,
    ))
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        title="3D Isosurface"
    )
    fig.update_scenes(
        camera_projection_type="orthographic",
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False
    )

    if filename:
        if filename.endswith('.html'):
            fig.write_html(filename)
        else:
            fig.write_image(filename)
        print(f"Saved: {filename}")
    else:
        fig.show()


def save_mat(filename, phi_dict, nx, lx):
    """Save fields as MATLAB .mat file for use with tools/.

    The saved file can be opened with:
    - tools/plot_2d_density.ipynb (2D)
    - tools/plot_3d_isodensity.ipynb (3D)
    """
    try:
        import scipy.io as sio
    except ImportError:
        print("scipy not installed. Install with: pip install scipy")
        return

    data = {
        'nx': np.array(nx),
        'lx': np.array(lx),
    }
    for name, phi in phi_dict.items():
        data[f'phi_{name}'] = phi.flatten()

    sio.savemat(filename, data)
    print(f"Saved: {filename}")
    print(f"Open with tools/plot_2d_density.ipynb or tools/plot_3d_isodensity.ipynb")


def save_vtk(filename, phi, lx):
    """Save 3D field as VTK file for ParaView."""
    nx, ny, nz = phi.shape
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Polymer concentration field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN 0 0 0\n")
        f.write(f"SPACING {lx[0]/nx} {lx[1]/ny} {lx[2]/nz}\n")
        f.write(f"POINT_DATA {nx*ny*nz}\n")
        f.write("SCALARS phi float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{phi[i,j,k]:.6f}\n")
    print(f"Saved VTK file: {filename}")
    print(f"Open with: paraview {filename}")


# =============================================================================
# Example: Run SCFT and visualize
# =============================================================================

if __name__ == "__main__":
    # Simple 2D cylinder example
    params = {
        "nx": [64, 64],
        "lx": [4.0, 4.0*np.sqrt(3)/2],  # Hexagonal aspect ratio

        "chain_model": "continuous",
        "ds": 1/100,

        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20},

        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.3},
                {"type": "B", "length": 0.7},
            ],
        }],

        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.02,
            "mix_init": 0.02,
        },

        "box_is_altering": False,

        "max_iter": 200,
        "tolerance": 1e-6,
        "verbose_level": 1,
    }

    # Initialize with cylinder seed
    nx, ny = params["nx"]
    w_A = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            x = i/nx - 0.5
            y = j/ny - 0.5
            r = np.sqrt(x**2 + y**2)
            w_A[i, j] = -0.5 * (1 - np.tanh((r - 0.15) / 0.05))
    w_A = w_A * params["chi_n"]["A,B"] * 0.5

    print("Running SCFT simulation...")
    calc = scft.SCFT(params=params)
    calc.run(initial_fields={"A": w_A, "B": -w_A})

    print(f"\nFree energy: {calc.free_energy:.6f}")

    # Get results
    phi_A = calc.phi["A"].reshape(params["nx"])
    phi_B = calc.phi["B"].reshape(params["nx"])
    lx = calc.prop_solver.get_lx()

    print("\n--- Visualization ---")

    # 1. Save as matplotlib figure
    visualize_2d({"A": phi_A, "B": phi_B}, lx, filename="cylinder_2d.png")

    # 2. Save as .mat file for use with tools/
    save_mat("cylinder_2d.mat", {"A": phi_A, "B": phi_B}, params["nx"], lx)

    print("\nVisualization complete!")
    print("\nFor interactive 3D visualization, see:")
    print("  - tools/plot_3d_isodensity.ipynb (uses plotly)")
