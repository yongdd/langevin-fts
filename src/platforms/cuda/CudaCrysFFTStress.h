/**
 * @file CudaCrysFFTStress.h
 * @brief Shared CrysFFT stress fast path for CUDA pseudo-spectral solvers.
 *
 * Mirrors the CPU CrysFFT stress computation in
 * CpuSolverPseudoBase::compute_single_segment_stress:
 *
 * 1. Fill the CrysFFT physical grid from the reduced basis (q2)
 * 2. Apply the per-axis k-space multiplier
 *    - continuous chains: k_d²
 *    - discrete chains:   exp(-k²·b²·ds/6)·k_d²
 * 3. Reduce back to the reduced basis
 * 4. Accumulate the orbit-weighted dot product with q1
 * 5. Symmetrize equivalent axes by crystal system (Cubic: x,y,z;
 *    Tetragonal: x,y) and scale: V_dd = b²·M_full·sum_d / L_d²
 *
 * Only the diagonal components are computed (the branch requires an
 * orthogonal box; cross-term slots are never consumed by the caller).
 *
 * All GPU work is enqueued on the caller-provided stream; the per-stream
 * CrysFFT object and work buffers must not be shared across streams.
 */

#ifndef CUDA_CRYS_FFT_STRESS_H_
#define CUDA_CRYS_FFT_STRESS_H_

#include <cstddef>
#include <cuda_runtime.h>

#include "CudaCrysFFT.h"

class SpaceGroup;

/**
 * @brief Inputs for the CrysFFT stress fast path (one stream's resources).
 */
struct CudaCrysFFTStressArgs
{
    CudaCrysFFTBase* crysfft;        ///< Per-stream CrysFFT object
    CudaCrysFFTMode mode;            ///< PmmmDct or Recursive3m
    bool identity_map;               ///< True when reduced basis matches physical grid
    int M_phys;                      ///< CrysFFT physical grid size
    int n_basis;                     ///< Number of reduced-basis points
    int M_full;                      ///< Full logical grid size
    const int* d_phys_to_reduced;    ///< Device map phys -> reduced (nullptr when identity)
    const int* d_reduced_to_phys;    ///< Device map reduced -> phys (nullptr when identity)
    const int* d_orbit_counts;       ///< Device orbit counts (size n_basis)
    double* d_phys_work;             ///< Device work buffer (size M_phys)
    double* d_phys_out;              ///< Device work buffer (size M_phys)
    double* d_reduce_buf;            ///< Device work buffer (size >= n_basis)
    void* d_cub_temp;                ///< CUB temporary storage
    size_t cub_temp_bytes;           ///< CUB temporary storage size
    const SpaceGroup* space_group;   ///< Space group (for crystal system)
    double lx[3];                    ///< Box lengths
    double bond_length_sq;           ///< b² (already halved for half-bond steps)
    bool include_bond_factor;        ///< True for discrete chains (exp(-k²·coeff) factor)
    double global_ds;                ///< Global contour step (discrete chains)
    cudaStream_t stream;             ///< Stream for all GPU work
};

/**
 * @brief Compute diagonal stress sums V₁₁, V₂₂, V₃₃ into d_segment_stress[0..2].
 *
 * The result matches the standard (full-grid cuFFT) stress path to machine
 * precision: sums are in deformation-vector units (divided by L_d²) and
 * scaled by b²·M_full, with equivalent-axis averaging by crystal system.
 *
 * @param args             Per-stream resources and physics parameters
 * @param d_q1_reduced     First propagator in reduced basis (device, size n_basis)
 * @param d_q2_reduced     Second propagator in reduced basis (device, size n_basis)
 * @param d_segment_stress Output device array (at least 3 doubles)
 */
void cuda_crysfft_compute_single_segment_stress(
    const CudaCrysFFTStressArgs& args,
    const double* d_q1_reduced,
    const double* d_q2_reduced,
    double* d_segment_stress);

#endif  // CUDA_CRYS_FFT_STRESS_H_
