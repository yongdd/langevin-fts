#!/usr/bin/env python3
"""
Generate reference values for TestRealSpaceLinearContinuous3D.

Run this script after building with 2nd-order mode (default) to get
the correct reference values for the test.

Usage:
    cd build && make -j8 && make install
    cd ../tests
    python generate_realspace_reference.py
"""

import numpy as np
import sys
sys.path.insert(0, '../build')

try:
    from polymerfts import PropagatorSolver
except ImportError:
    print("Error: polymerfts not found. Please build and install first.")
    print("  cd build && make -j8 && make install")
    sys.exit(1)

# Test parameters matching TestRealSpaceLinearContinuous3D.cpp
II, JJ, KK = 5, 4, 3
M = II * JJ * KK
NN = 4
f = 0.5
Lx, Ly, Lz = 4.0, 3.0, 2.0

w_a = np.array([
    0.183471406e+0,0.623968915e+0,0.731257661e+0,0.997228140e+0,0.961913696e+0,
    0.792673860e-1,0.429684069e+0,0.290531312e+0,0.453270921e+0,0.199228629e+0,
    0.754931905e-1,0.226924328e+0,0.936407886e+0,0.979392715e+0,0.464957186e+0,
    0.742653949e+0,0.368019859e+0,0.885231224e+0,0.406191773e+0,0.653096157e+0,
    0.567929080e-1,0.568028857e+0,0.144986181e+0,0.466158777e+0,0.573327733e+0,
    0.136324723e+0,0.819010407e+0,0.271218167e+0,0.626224101e+0,0.398109186e-1,
    0.860031651e+0,0.338153865e+0,0.688078522e+0,0.564682952e+0,0.222924187e+0,
    0.306816449e+0,0.316316038e+0,0.640568415e+0,0.702342408e+0,0.632135481e+0,
    0.649402777e+0,0.647100865e+0,0.370402133e+0,0.691313864e+0,0.447870566e+0,
    0.757298851e+0,0.586173682e+0,0.766745717e-1,0.504185402e+0,0.812016428e+0,
    0.217988206e+0,0.273487202e+0,0.937672578e+0,0.570540523e+0,0.409071185e+0,
    0.391548274e-1,0.663478965e+0,0.260755447e+0,0.503943226e+0,0.979481790e+0
])

w_b = np.array([
    0.113822903e-1,0.330673934e+0,0.270138412e+0,0.669606774e+0,0.885344778e-1,
    0.604752856e+0,0.890062293e+0,0.328557615e+0,0.965824739e+0,0.865399960e+0,
    0.698893686e+0,0.857947305e+0,0.594897904e+0,0.248187208e+0,0.155686710e+0,
    0.116803898e+0,0.711146609e+0,0.107610460e+0,0.143034307e+0,0.123131521e+0,
    0.230387237e+0,0.516274641e+0,0.562366089e-1,0.491449746e+0,0.746656140e+0,
    0.296108614e+0,0.424987667e+0,0.651538750e+0,0.116745920e+0,0.567790110e+0,
    0.954487190e+0,0.802476927e-1,0.440223916e+0,0.843025420e+0,0.612864528e+0,
    0.571893767e+0,0.759625605e+0,0.872255004e+0,0.935065364e+0,0.635565347e+0,
    0.373711972e-2,0.860683468e+0,0.186492706e+0,0.267880995e+0,0.579305501e+0,
    0.693549226e+0,0.613843845e+0,0.259811620e-1,0.848915465e+0,0.766111508e+0,
    0.872008750e+0,0.116289041e+0,0.917713893e+0,0.710076955e+0,0.442712526e+0,
    0.516722213e+0,0.253395805e+0,0.472950065e-1,0.152934959e+0,0.292486174e+0
])

def format_array(arr, name, cols=3):
    """Format array as C++ initializer."""
    print(f"        double {name}[M] =")
    print("        {")
    for i in range(0, len(arr), cols):
        row = arr[i:i+cols]
        line = ", ".join(f"{v:.10e}" for v in row)
        if i + cols < len(arr):
            line += ","
        print(f"            {line}")
    print("        };")

def main():
    print("Generating reference values for TestRealSpaceLinearContinuous3D")
    print("=" * 60)

    # Create solver
    solver = PropagatorSolver(
        nx=[II, JJ, KK],
        lx=[Lx, Ly, Lz],
        ds=1.0/NN,
        bond_lengths={"A": 1.0, "B": 1.0},
        bc=["periodic", "periodic", "periodic", "periodic", "periodic", "periodic"],
        chain_model="continuous",
        method="realspace",
        platform="cpu-mkl",
        reduce_memory_usage=False
    )

    # Add polymer
    solver.add_polymer(
        volume_fraction=1.0,
        blocks=[["A", f, 0, 1], ["B", 1.0-f, 1, 2]],
        grafting_points={}
    )

    # Compute propagators
    solver.compute_propagators({"A": w_a, "B": w_b})
    solver.compute_concentrations()

    # Get results
    q1_last = solver.get_propagator(polymer=0, v=1, u=2, step=2)
    q2_last = solver.get_propagator(polymer=0, v=1, u=0, step=2)
    phi_a = solver.get_concentration("A")
    phi_b = solver.get_concentration("B")
    QQ = solver.get_partition_function(polymer=0)

    print("\n// 2nd-order Crank-Nicolson reference values")
    format_array(q1_last, "q1_last_ref")
    print()
    format_array(q2_last, "q2_last_ref")
    print()
    format_array(phi_a, "phi_a_ref")
    print()
    format_array(phi_b, "phi_b_ref")
    print()
    print(f"        double QQ_ref = {QQ:.11f};")
    print()
    print("=" * 60)
    print("Copy the above values into TestRealSpaceLinearContinuous3D.cpp")
    print("in the #else section (2nd-order reference values)")

if __name__ == "__main__":
    main()
