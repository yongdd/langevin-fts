#!/usr/bin/env python3
"""
Test space group SCFT phases against expected free energies from README.md.

This test verifies that SCFT with space group symmetry converges to the
correct free energy for all 14 crystallographic phases.

Phases tested:
- Cubic: BCC, FCC, SC, A15
- Network (from .mat): DG, DD, DP, SD, SG, SP
- Tetragonal: Sigma
- Orthorhombic: Fddd
- Hexagonal: HCP, PL
"""

import os
import sys
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter, zoom

# Set environment before importing polymerfts
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import scft
from polymerfts import _core

# Path to phase data files
PHASES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "scft", "phases")

# Tolerance for free energy comparison
FE_TOLERANCE = 1e-5


def get_platform():
    """Get available platform (cuda if available, otherwise cpu-fftw)."""
    available = _core.PlatformSelector.avail_platforms()
    if "cuda" in available:
        return "cuda"
    return "cpu-fftw"


def run_scft_phase(params, w_init):
    """Run SCFT and return free energy."""
    calc = scft.SCFT(params=params)
    calc.run(initial_fields=w_init)
    return calc.free_energy, calc.error_level


# =============================================================================
# Cubic Phases
# =============================================================================

def test_bcc():
    """Test BCC phase (Im-3m). Expected FE: -0.0899204"""
    print("Testing BCC (Im-3m)...")

    f = 24/90
    nx = [32, 32, 32]
    lx = [1.9, 1.9, 1.9]
    expected_fe = -0.0899204

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 18.1},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Im-3m", "number": 529},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"BCC free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_fcc():
    """Test FCC phase (Fm-3m). Expected FE: -0.0889892"""
    print("Testing FCC (Fm-3m)...")

    f = 24/90
    nx = [32, 32, 32]
    lx = [1.91, 1.91, 1.91]
    expected_fe = -0.0889892

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 18.1},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Fm-3m", "number": 523},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    fcc_positions = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    for x, y, z in fcc_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"FCC free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_sc():
    """Test SC phase (Pm-3m). Expected FE: -0.1218046"""
    print("Testing SC (Pm-3m)...")

    f = 0.2
    nx = [32, 32, 32]
    lx = [1.5, 1.5, 1.5]
    expected_fe = -0.1218046

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "stress_interval": 1,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 25},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Pm-3m", "number": 517},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8, "verbose": 0,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[0, 0, 0]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"SC free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_a15():
    """Test A15 phase (Pm-3n). Expected FE: -0.8685845"""
    print("Testing A15 (Pm-3n)...")

    f = 0.3
    eps = 2.0
    nx = [64, 64, 64]
    lx = [4.0, 4.0, 4.0]
    expected_fe = -0.8685845

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {
            "A": np.sqrt(eps*eps/(eps*eps*f + (1-f))),
            "B": np.sqrt(1.0/(eps*eps*f + (1-f))),
        },
        "chi_n": {"A,B": 25},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Pm-3n", "number": 520},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[0,0,0],[1/2,1/2,1/2],
        [1/4,1/2,0],[3/4,1/2,0],[1/2,0,1/4],[1/2,0,3/4],[0,1/4,1/2],[0,3/4,1/2]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"A15 free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


# =============================================================================
# Network Phases (from .mat files)
# =============================================================================

def test_gyroid():
    """Test Gyroid/DG phase (Ia-3d). Expected FE: -0.2131824"""
    print("Testing Gyroid/DG (Ia-3d)...")

    mat_file = os.path.join(PHASES_DIR, "DG.mat")
    if not os.path.exists(mat_file):
        print("  SKIPPED (DG.mat not found)")
        return

    f = 0.4
    nx = [32, 32, 32]
    expected_fe = -0.2131824

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    lx = list(input_data["lx"])
    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(nx)/input_data["nx"])

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Ia-3d", "number": 530},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"Gyroid free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_dd():
    """Test Double Diamond phase (Pn-3m). Expected FE: -0.1997210"""
    print("Testing DD (Pn-3m)...")

    mat_file = os.path.join(PHASES_DIR, "DD.mat")
    if not os.path.exists(mat_file):
        print("  SKIPPED (DD.mat not found)")
        return

    f = 0.4
    nx = [48, 48, 48]
    expected_fe = -0.1997210

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    lx = list(input_data["lx"])
    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(nx)/input_data["nx"])

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Pn-3m", "number": 522},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"DD free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_dp():
    """Test Double Primitive phase (Im-3m). Expected FE: -0.1445778"""
    print("Testing DP (Im-3m)...")

    mat_file = os.path.join(PHASES_DIR, "DP.mat")
    if not os.path.exists(mat_file):
        print("  SKIPPED (DP.mat not found)")
        return

    f = 0.4
    nx = [32, 32, 32]
    expected_fe = -0.1445778

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    lx = list(input_data["lx"])
    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(nx)/input_data["nx"])

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Im-3m", "number": 529},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"DP free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_sd():
    """Test Single Diamond phase (Fd-3m). Expected FE: -0.1927212"""
    print("Testing SD (Fd-3m)...")

    mat_file = os.path.join(PHASES_DIR, "SD.mat")
    if not os.path.exists(mat_file):
        print("  SKIPPED (SD.mat not found)")
        return

    f = 0.4
    nx = [32, 32, 32]
    expected_fe = -0.1927212

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    lx = list(input_data["lx"])
    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(nx)/input_data["nx"])

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Fd-3m", "number": 526},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"SD free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_sg():
    """Test Single Gyroid phase (I4_132). Expected FE: -0.1973548"""
    print("Testing SG (I4_132)...")

    mat_file = os.path.join(PHASES_DIR, "SG.mat")
    if not os.path.exists(mat_file):
        print("  SKIPPED (SG.mat not found)")
        return

    f = 0.4
    nx = [32, 32, 32]
    expected_fe = -0.1973548

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    lx = list(input_data["lx"])
    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(nx)/input_data["nx"])

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "I4_132", "number": 510},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"SG free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_sp():
    """Test Single Primitive phase (Pm-3m). Expected FE: -0.1735695"""
    print("Testing SP (Pm-3m)...")

    mat_file = os.path.join(PHASES_DIR, "SP.mat")
    if not os.path.exists(mat_file):
        print("  SKIPPED (SP.mat not found)")
        return

    f = 0.4
    nx = [32, 32, 32]
    expected_fe = -0.1735695

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    lx = list(input_data["lx"])
    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(nx)/input_data["nx"])

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Pm-3m", "number": 517},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"SP free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


# =============================================================================
# Tetragonal Phase
# =============================================================================

def test_sigma():
    """Test Sigma phase (P4_2/mnm). Expected FE: -0.4695150"""
    print("Testing Sigma (P4_2/mnm)...")

    f = 0.25
    eps = 2.0
    nx = [64, 64, 32]
    lx = [7.0, 7.0, 4.0]
    expected_fe = -0.4695150

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {
            "A": np.sqrt(eps*eps/(eps*eps*f + (1-f))),
            "B": np.sqrt(1.0/(eps*eps*f + (1-f))),
        },
        "chi_n": {"A,B": 25},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "B", "length": 1-f},
                {"type": "A", "length": f},
            ],
        }],
        "space_group": {"symbol": "P4_2/mnm", "number": 419},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.50],
        [0.40, 0.40, 0.00], [0.60, 0.60, 0.00], [0.90, 0.10, 0.50], [0.10, 0.90, 0.50],
        [0.46, 0.13, 0.00], [0.13, 0.46, 0.00], [0.87, 0.54, 0.00], [0.54, 0.87, 0.00],
        [0.63, 0.04, 0.50], [0.04, 0.63, 0.50], [0.96, 0.37, 0.50], [0.37, 0.96, 0.50],
        [0.74, 0.07, 0.00], [0.07, 0.74, 0.00], [0.93, 0.26, 0.00], [0.26, 0.93, 0.00],
        [0.43, 0.24, 0.50], [0.24, 0.43, 0.50], [0.76, 0.57, 0.50], [0.56, 0.77, 0.50],
        [0.18, 0.18, 0.25], [0.82, 0.82, 0.25], [0.68, 0.32, 0.25], [0.32, 0.68, 0.25],
        [0.18, 0.18, 0.75], [0.82, 0.82, 0.75], [0.68, 0.32, 0.75], [0.32, 0.68, 0.75]
    ]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"Sigma free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


# =============================================================================
# Orthorhombic Phase
# =============================================================================

def test_fddd():
    """Test Fddd/O70 phase (Fddd). Expected FE: -0.1606698"""
    print("Testing Fddd (Fddd)...")

    mat_file = os.path.join(PHASES_DIR, "FdddInput.mat")
    if not os.path.exists(mat_file):
        print("  SKIPPED (FdddInput.mat not found)")
        return

    f = 0.43
    nx = [84, 48, 24]
    lx = [5.58, 3.17, 1.59]
    expected_fe = -0.1606698

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(nx)/input_data["nx"])

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "stress_interval": 1,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 14.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Fddd", "number": 336},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8, "verbose": 0,
    }

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"Fddd free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


# =============================================================================
# Hexagonal Phases
# =============================================================================

def test_hcp():
    """Test HCP phase (P6_3/mmc). Expected FE: -0.1345346"""
    print("Testing HCP (P6_3/mmc)...")

    f = 0.25
    nx = [24, 24, 24]
    lx = [1.7186, 1.7186, 2.7982]
    expected_fe = -0.1345346

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "angles": [90.0, 90.0, 120.0],
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "P6_3/mmc", "number": 488},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"HCP free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


def test_pl():
    """Test PL/HPL phase (P6/mmm). Expected FE: -0.2119551"""
    print("Testing PL (P6/mmm)...")

    f = 0.4
    nx = [24, 24, 36]
    lx = [1.958, 1.958, 2.981]
    expected_fe = -0.2119551

    params = {
        "platform": get_platform(),
        "nx": nx, "lx": lx,
        "angles": [90.0, 90.0, 120.0],
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "P6/mmm", "number": 485},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 1000, "tolerance": 1e-8, "verbose": 0,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.0], [2/3, 1/3, 0.0],
        [1/3, 2/3, 0.5], [2/3, 1/3, 0.5],
    ]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
    diff = abs(fe - expected_fe)
    print(f"  Free energy: {fe:.7f}, Expected: {expected_fe:.7f}, Diff: {diff:.2e}")
    assert diff < FE_TOLERANCE, f"PL free energy mismatch: {fe} vs {expected_fe}"
    print("  PASSED")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Space Group Phase Free Energy Tests (14 phases)")
    print("=" * 60 + "\n")

    try:
        # Cubic phases (4)
        test_bcc()
        test_fcc()
        test_sc()
        test_a15()

        # Network phases from .mat files (6)
        test_gyroid()
        test_dd()
        test_dp()
        test_sd()
        test_sg()
        test_sp()

        # Tetragonal (1)
        test_sigma()

        # Orthorhombic (1)
        test_fddd()

        # Hexagonal (2)
        test_hcp()
        test_pl()

        print("\n" + "=" * 60)
        print("All 14 tests PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
