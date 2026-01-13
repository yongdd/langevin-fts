#!/usr/bin/env python3
"""Analyze Fig 1 benchmark results for all numerical methods."""

import json
import glob
import os
from collections import defaultdict

def load_results():
    """Load all benchmark results."""
    results = defaultdict(dict)

    for filepath in glob.glob("benchmark_fig1_*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)

        method = data['metadata']['method']
        for r in data['results']:
            if r.get('success', False):
                Ns = r['Ns']
                results[method][Ns] = {
                    'free_energy': r['free_energy'],
                    'run_time_s': r['run_time_s'],
                    'total_time_s': r['total_time_s'],
                    'converged': r['converged'],
                    'error': r['final_error'],
                }

    return results

def print_summary(results):
    """Print summary tables."""
    methods = ['rqm4', 'etdrk4', 'cn-adi2', 'cn-adi4']

    # Get all Ns values
    all_ns = set()
    for method in methods:
        if method in results:
            all_ns.update(results[method].keys())
    all_ns = sorted(all_ns)

    print("="*80)
    print("FREE ENERGY vs Ns")
    print("="*80)
    print(f"{'Method':<12}", end="")
    for ns in all_ns:
        print(f"{ns:>12}", end="")
    print()
    print("-"*80)

    for method in methods:
        if method in results:
            print(f"{method.upper():<12}", end="")
            for ns in all_ns:
                if ns in results[method]:
                    fe = results[method][ns]['free_energy']
                    print(f"{fe:>12.8f}", end="")
                else:
                    print(f"{'—':>12}", end="")
            print()

    print()
    print("="*80)
    print("RUN TIME (seconds) vs Ns")
    print("="*80)
    print(f"{'Method':<12}", end="")
    for ns in all_ns:
        print(f"{ns:>10}", end="")
    print()
    print("-"*80)

    for method in methods:
        if method in results:
            print(f"{method.upper():<12}", end="")
            for ns in all_ns:
                if ns in results[method]:
                    time = results[method][ns]['run_time_s']
                    print(f"{time:>10.1f}", end="")
                else:
                    print(f"{'—':>10}", end="")
            print()

    # Reference free energy (from highest Ns)
    ref_ns = max(all_ns)
    F_ref = results['rqm4'].get(ref_ns, {}).get('free_energy', -0.47697411)

    print()
    print("="*80)
    print(f"ERROR |F - F_ref| (F_ref = {F_ref:.10f} from Ns={ref_ns})")
    print("="*80)
    print(f"{'Method':<12}", end="")
    for ns in all_ns:
        print(f"{ns:>12}", end="")
    print()
    print("-"*80)

    for method in methods:
        if method in results:
            print(f"{method.upper():<12}", end="")
            for ns in all_ns:
                if ns in results[method]:
                    fe = results[method][ns]['free_energy']
                    err = abs(fe - F_ref)
                    print(f"{err:>12.2e}", end="")
                else:
                    print(f"{'—':>12}", end="")
            print()

    # Speedup relative to CN-ADI2 at Ns=1000
    print()
    print("="*80)
    print("SPEEDUP relative to CN-ADI2 at Ns=1000")
    print("="*80)

    ref_time = results.get('cn-adi2', {}).get(1000, {}).get('run_time_s', None)
    if ref_time:
        for method in methods:
            if method in results and 1000 in results[method]:
                time = results[method][1000]['run_time_s']
                speedup = ref_time / time
                print(f"{method.upper():<12}: {speedup:.2f}x (time = {time:.1f}s)")

    # Markdown tables for documentation
    print()
    print("="*80)
    print("MARKDOWN TABLES FOR DOCUMENTATION")
    print("="*80)

    # Execution time table
    ns_time = [100, 200, 400, 1000, 4000]
    print("\n### Execution Time vs Contour Steps (Ns)\n")
    print("| Method |", end="")
    for ns in ns_time:
        print(f" Ns={ns} |", end="")
    print()
    print("|--------|" + "--------|" * len(ns_time))

    for method in methods:
        if method in results:
            print(f"| **{method.upper()}** |", end="")
            for ns in ns_time:
                if ns in results[method]:
                    time = results[method][ns]['run_time_s']
                    print(f" {time:.1f} s |", end="")
                else:
                    print(" — |", end="")
            print()

    # Free energy table
    ns_fe = [40, 80, 160, 320, 640, 1000]
    print("\n### Free Energy vs Contour Steps (Ns)\n")
    print("| Method |", end="")
    for ns in ns_fe:
        print(f" Ns={ns} |", end="")
    print()
    print("|--------|" + "--------|" * len(ns_fe))

    for method in methods:
        if method in results:
            print(f"| **{method.upper()}** |", end="")
            for ns in ns_fe:
                if ns in results[method]:
                    fe = results[method][ns]['free_energy']
                    print(f" {fe:.8f} |", end="")
                else:
                    print(" — |", end="")
            print()

if __name__ == "__main__":
    os.chdir("/home/yongdd/polymer/langevin-fts/tests")
    results = load_results()
    print_summary(results)
