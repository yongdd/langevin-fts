#!/usr/bin/env python3
"""
Test space group SCFT phases against expected free energies.

This test verifies that SCFT with space group symmetry converges to the
correct free energy for crystallographic phases.

Tests all combinations of:
- Platform: all available platforms (cuda, cpu-fftw)
- Memory saving: reduce_memory=True/False

Uses pre-converged .mat files from tests/data/ for fast iteration (~1-2 iterations).
"""

import os
import sys
import time
import numpy as np
import scipy.io
from scipy.ndimage import zoom

# Set environment before importing polymerfts
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import scft
from polymerfts import _core

# Path to pre-converged phase data files
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Tolerance for free energy comparison
FE_TOLERANCE = 1e-10


def get_available_platforms():
    """Get list of available platforms."""
    return _core.PlatformSelector.avail_platforms()


def load_mat_fields(mat_file, target_nx):
    """Load and resize fields from .mat file to target grid.

    Returns (lx, w_A, w_B) or (None, None, None) if file has invalid data.
    """
    if not os.path.exists(mat_file):
        return None, None, None

    input_data = scipy.io.loadmat(mat_file, squeeze_me=True)
    lx = list(input_data["lx"])

    # Check for NaN values
    if np.isnan(lx).any() or np.isnan(input_data["w_A"]).any():
        return None, None, None

    w_A = zoom(np.reshape(input_data["w_A"], input_data["nx"]), np.array(target_nx)/input_data["nx"])
    w_B = zoom(np.reshape(input_data["w_B"], input_data["nx"]), np.array(target_nx)/input_data["nx"])
    return lx, w_A, w_B


def run_scft_phase(params, w_init):
    """Run SCFT and return free energy."""
    calc = scft.SCFT(params=params)
    calc.run(initial_fields=w_init)
    return calc.free_energy, calc.error_level


# =============================================================================
# Phase Definitions with 15-digit precision free energies
# =============================================================================

PHASES = {
    "BCC": {
        "mat_file": "BCC.mat",
        "f": 24/90,
        "nx": [32, 32, 32],
        "expected_fe": -0.089919810968570,
        "space_group": {"symbol": "Im-3m", "number": 529},
        "ds": 1/90,
        "chi_n": {"A,B": 18.1},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "FCC": {
        "mat_file": "FCC.mat",
        "f": 24/90,
        "nx": [32, 32, 32],
        "expected_fe": -0.088943830493689,
        "space_group": {"symbol": "Fm-3m", "number": 523},
        "ds": 1/90,
        "chi_n": {"A,B": 18.1},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "SC": {
        "mat_file": "SC.mat",
        "f": 0.2,
        "nx": [32, 32, 32],
        "expected_fe": -0.121829515633899,
        "space_group": {"symbol": "Pm-3m", "number": 517},
        "ds": 1/100,
        "chi_n": {"A,B": 25},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "A15": {
        "mat_file": "A15.mat",
        "f": 0.3,
        "nx": [64, 64, 64],
        "expected_fe": -0.868810149461849,
        "space_group": {"symbol": "Pm-3n", "number": 520},
        "ds": 1/100,
        "chi_n": {"A,B": 25},
        "eps": 2.0,  # conformational asymmetry
    },
    "DG": {
        "mat_file": "DG.mat",
        "f": 0.4,
        "nx": [32, 32, 32],
        "expected_fe": -0.212985751633016,
        "space_group": {"symbol": "Ia-3d", "number": 530},
        "ds": 1/100,
        "chi_n": {"A,B": 15},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "DD": {
        "mat_file": "DD.mat",
        "f": 0.4,
        "nx": [48, 48, 48],
        "expected_fe": -0.199722617070025,
        "space_group": {"symbol": "Pn-3m", "number": 522},
        "ds": 1/100,
        "chi_n": {"A,B": 15},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "DP": {
        "mat_file": "DP.mat",
        "f": 0.4,
        "nx": [48, 48, 48],
        "expected_fe": -0.144579007615457,
        "space_group": {"symbol": "Im-3m", "number": 529},
        "ds": 1/100,
        "chi_n": {"A,B": 15},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "SD": {
        "mat_file": "SD.mat",
        "f": 0.4,
        "nx": [48, 48, 48],
        "expected_fe": -0.192723306339966,
        "space_group": {"symbol": "Fd-3m", "number": 526},
        "ds": 1/100,
        "chi_n": {"A,B": 15},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "SG": {
        "mat_file": "SG.mat",
        "f": 0.4,
        "nx": [32, 32, 32],
        "expected_fe": -0.197379170734585,
        "space_group": {"symbol": "I4_132", "number": 510},
        "ds": 1/100,
        "chi_n": {"A,B": 15},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "SP": {
        "mat_file": "SP.mat",
        "f": 0.4,
        "nx": [32, 32, 32],
        "expected_fe": -0.173571138977048,
        "space_group": {"symbol": "Pm-3m", "number": 517},
        "ds": 1/100,
        "chi_n": {"A,B": 15},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "Sigma": {
        "mat_file": "Sigma.mat",
        "f": 0.25,
        "nx": [64, 64, 32],
        "expected_fe": -0.470356276886735,
        "space_group": {"symbol": "P4_2/mnm", "number": 419},
        "ds": 1/100,
        "chi_n": {"A,B": 25},
        "eps": 2.0,
        "block_order": "BA",  # BA block order
    },
    "Fddd": {
        "mat_file": "Fddd.mat",
        "f": 0.43,
        "nx": [84, 48, 24],
        "expected_fe": -0.160677247809990,
        "space_group": {"symbol": "Fddd", "number": 336},
        "ds": 1/100,
        "chi_n": {"A,B": 14.0},
        "segment_lengths": {"A": 1.0, "B": 1.0},
    },
    "HCP": {
        "mat_file": "HCP.mat",
        "f": 0.25,
        "nx": [48, 48, 48],
        "expected_fe": -0.133556310465600,
        "space_group": {"symbol": "P6_3/mmc", "number": 488},
        "ds": 1/100,
        "chi_n": {"A,B": 20},
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "angles": [90.0, 90.0, 120.0],
    },
    "PL": {
        "mat_file": "PL.mat",
        "f": 0.4,
        "nx": [48, 48, 72],
        "expected_fe": -0.211955441747189,
        "space_group": {"symbol": "P6/mmm", "number": 485},
        "ds": 1/100,
        "chi_n": {"A,B": 15},
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "angles": [90.0, 90.0, 120.0],
    },
}


def build_params(phase_def, lx, platform, reduce_memory):
    """Build SCFT params from phase definition."""
    f = phase_def["f"]

    # Handle conformational asymmetry
    if "eps" in phase_def:
        eps = phase_def["eps"]
        segment_lengths = {
            "A": np.sqrt(eps*eps/(eps*eps*f + (1-f))),
            "B": np.sqrt(1.0/(eps*eps*f + (1-f))),
        }
    else:
        segment_lengths = phase_def["segment_lengths"]

    # Handle block order (BA for Sigma)
    if phase_def.get("block_order") == "BA":
        blocks = [
            {"type": "B", "length": 1-f},
            {"type": "A", "length": f},
        ]
    else:
        blocks = [
            {"type": "A", "length": f},
            {"type": "B", "length": 1-f},
        ]

    params = {
        "platform": platform,
        "nx": phase_def["nx"],
        "lx": lx,
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": phase_def["ds"],
        "segment_lengths": segment_lengths,
        "chi_n": phase_def["chi_n"],
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": blocks,
        }],
        "space_group": phase_def["space_group"],
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 5,
        "tolerance": 1e-8,
        "verbose": 0,
        "reduce_memory": reduce_memory,
    }

    # Add angles for hexagonal phases
    if "angles" in phase_def:
        params["angles"] = phase_def["angles"]

    return params


def test_phase(phase_name, platform, reduce_memory):
    """Test a single phase with given platform and memory option.

    Returns (status, fe, msg, elapsed_time)
    """
    phase_def = PHASES[phase_name]
    mat_file = os.path.join(DATA_DIR, phase_def["mat_file"])

    lx, w_A, w_B = load_mat_fields(mat_file, phase_def["nx"])
    if lx is None:
        return "SKIPPED", None, f"{phase_def['mat_file']} not found or invalid", 0.0

    params = build_params(phase_def, lx, platform, reduce_memory)

    try:
        t_start = time.perf_counter()
        fe, error = run_scft_phase(params, {"A": w_A.flatten(), "B": w_B.flatten()})
        elapsed = time.perf_counter() - t_start
        diff = abs(fe - phase_def["expected_fe"])

        if diff < FE_TOLERANCE:
            return "PASSED", fe, f"FE={fe:.15f}, diff={diff:.2e}, time={elapsed:.3f}s", elapsed
        else:
            return "FAILED", fe, f"FE mismatch: {fe:.15f} vs {phase_def['expected_fe']:.15f}, diff={diff:.2e}", elapsed
    except Exception as e:
        return "ERROR", None, str(e), 0.0


def main():
    print("=" * 70)
    print("Space Group Phase Tests - All Platforms and Memory Options")
    print("=" * 70 + "\n")

    available_platforms = get_available_platforms()
    print(f"Available platforms: {available_platforms}\n")

    # Test configurations: use only available platforms
    configs = []
    for platform in available_platforms:
        configs.append((platform, False))  # Normal mode
        configs.append((platform, True))   # Memory-saving mode

    phase_names = list(PHASES.keys())

    # Results storage
    results = {config: {} for config in configs}
    times = {config: {} for config in configs}
    fe_diffs = {config: {} for config in configs}

    # Run tests
    for platform, reduce_memory in configs:
        mode_str = "reduce_memory" if reduce_memory else "normal"
        print(f"\n{'='*70}")
        print(f"Platform: {platform}, Mode: {mode_str}")
        print("=" * 70)

        for phase_name in phase_names:
            status, fe, msg, elapsed = test_phase(phase_name, platform, reduce_memory)
            results[(platform, reduce_memory)][phase_name] = status
            times[(platform, reduce_memory)][phase_name] = elapsed
            if fe is not None:
                fe_diffs[(platform, reduce_memory)][phase_name] = abs(fe - PHASES[phase_name]["expected_fe"])
            else:
                fe_diffs[(platform, reduce_memory)][phase_name] = None

            status_symbol = {"PASSED": "✓", "FAILED": "✗", "SKIPPED": "-", "ERROR": "!"}[status]
            print(f"  [{status_symbol}] {phase_name:8s}: {msg}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Results")
    print("=" * 70)

    # Header
    header = f"{'Phase':10s}"
    for platform, reduce_memory in configs:
        mode_str = "mem" if reduce_memory else "std"
        header += f" | {platform[:4]}/{mode_str}"
    print(header)
    print("-" * len(header))

    # Results per phase
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for phase_name in phase_names:
        row = f"{phase_name:10s}"
        for config in configs:
            status = results[config].get(phase_name, "N/A")
            symbol = {"PASSED": "✓", "FAILED": "✗", "SKIPPED": "-", "ERROR": "!", "N/A": "?"}[status]
            row += f" |    {symbol}    "
            if status == "PASSED":
                total_passed += 1
            elif status == "FAILED" or status == "ERROR":
                total_failed += 1
            else:
                total_skipped += 1
        print(row)

    print("-" * len(header))
    print(f"\nTotal: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")

    # Free energy difference summary
    print("\n" + "=" * 70)
    print("SUMMARY - Free Energy Difference")
    print("=" * 70)

    # FE diff header
    fe_header = f"{'Phase':10s}"
    for platform, reduce_memory in configs:
        mode_str = "mem" if reduce_memory else "std"
        fe_header += f" | {platform[:4]}/{mode_str:3s}"
    print(fe_header)
    print("-" * len(fe_header))

    # FE diff per phase
    for phase_name in phase_names:
        row = f"{phase_name:10s}"
        for config in configs:
            diff = fe_diffs[config].get(phase_name)
            if diff is not None:
                row += f" |  {diff:8.2e}"
            else:
                row += f" |     N/A   "
        print(row)

    print("-" * len(fe_header))

    # Timing summary
    print("\n" + "=" * 70)
    print("SUMMARY - Timing (seconds)")
    print("=" * 70)

    # Timing header
    time_header = f"{'Phase':10s}"
    for platform, reduce_memory in configs:
        mode_str = "mem" if reduce_memory else "std"
        time_header += f" | {platform[:4]}/{mode_str:3s}"
    print(time_header)
    print("-" * len(time_header))

    # Timing per phase
    total_times = {config: 0.0 for config in configs}
    for phase_name in phase_names:
        row = f"{phase_name:10s}"
        for config in configs:
            t = times[config].get(phase_name, 0.0)
            total_times[config] += t
            row += f" |   {t:6.2f}  "
        print(row)

    # Total time row
    print("-" * len(time_header))
    total_row = f"{'TOTAL':10s}"
    for config in configs:
        total_row += f" |   {total_times[config]:6.2f}  "
    print(total_row)

    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
