/**
 * @file CudaCrysFFTStress.cu
 * @brief Shared CrysFFT stress fast path for CUDA pseudo-spectral solvers.
 *
 * See CudaCrysFFTStress.h for the algorithm description. This mirrors the
 * CPU implementation in CpuSolverPseudoBase::compute_single_segment_stress.
 */

#include "CudaCrysFFTStress.h"

#include <string>

#include <cub/device/device_reduce.cuh>

#include "CudaCommon.h"
#include "CudaCrysFFTRecursive3m.h"
#include "Exception.h"
#include "SpaceGroup.h"

namespace {

/**
 * @brief Symmetrize equivalent axes and apply final scaling (single thread).
 *
 * Space groups whose point-group operations permute axes (cubic: x,y,z;
 * tetragonal: x,y) make the per-axis multipliers k_d² non-orbit-invariant,
 * so the reduced-basis accumulation scrambles the individual components
 * (only their sum is invariant). Since the kernel q1·q2 is space-group
 * symmetric, the exact per-direction sum is the average over equivalent axes.
 *
 * @param d_stress Device array holding raw sums in [0..2]
 * @param sym_mode 0: none, 1: tetragonal (avg x,y), 2: cubic (avg x,y,z)
 * @param s0,s1,s2 Final scale factors b²·M_full/L_d²
 */
__global__ void ker_crysfft_stress_finalize(
    double* d_stress, int sym_mode, double s0, double s1, double s2)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    double v0 = d_stress[0];
    double v1 = d_stress[1];
    double v2 = d_stress[2];

    if (sym_mode == 2)
    {
        double avg = (v0 + v1 + v2) / 3.0;
        v0 = avg;
        v1 = avg;
        v2 = avg;
    }
    else if (sym_mode == 1)
    {
        double avg = (v0 + v1) / 2.0;
        v0 = avg;
        v1 = avg;
    }

    d_stress[0] = v0 * s0;
    d_stress[1] = v1 * s1;
    d_stress[2] = v2 * s2;
}

}  // namespace

void cuda_crysfft_compute_single_segment_stress(
    const CudaCrysFFTStressArgs& args,
    const double* d_q1_reduced,
    const double* d_q2_reduced,
    double* d_segment_stress)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    if (args.mode != CudaCrysFFTMode::PmmmDct && args.mode != CudaCrysFFTMode::Recursive3m)
        throw_with_line_number("CrysFFT stress fast path supports only Pmmm and Recursive3m modes.");

    // Bond factor coefficient (discrete chains): b²·ds/6
    const double coeff = args.include_bond_factor ? args.bond_length_sq * args.global_ds / 6.0 : 0.0;

    for (int d = 0; d < 3; ++d)
    {
        // ===== Step 1: Fill physical grid from reduced basis (q2) =====
        const double* d_q2_phys;
        if (args.identity_map)
        {
            d_q2_phys = d_q2_reduced;
        }
        else
        {
            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
                args.d_phys_work, d_q2_reduced, args.d_phys_to_reduced, args.M_phys);
            gpu_error_check(cudaPeekAtLastError());
            d_q2_phys = args.d_phys_work;
        }

        // ===== Step 2: Apply per-axis multiplier (input is preserved) =====
        if (args.mode == CudaCrysFFTMode::PmmmDct)
        {
            CudaCrysFFT* pmmm = static_cast<CudaCrysFFT*>(args.crysfft);
            const double* d_mult = args.include_bond_factor
                ? pmmm->get_axis_boltz_k2_multiplier(d, coeff)
                : pmmm->get_axis_k2_multiplier(d);
            pmmm->apply_multiplier(const_cast<double*>(d_q2_phys), args.d_phys_out, d_mult, args.stream);
        }
        else  // Recursive3m
        {
            CudaCrysFFTRecursive3m* rec = static_cast<CudaCrysFFTRecursive3m*>(args.crysfft);
            using MT = CudaCrysFFTRecursive3m::MultiplierType;
            MT type;
            if (args.include_bond_factor)
                type = (d == 0) ? MT::ExpKx2 : (d == 1) ? MT::ExpKy2 : MT::ExpKz2;
            else
                type = (d == 0) ? MT::Kx2 : (d == 1) ? MT::Ky2 : MT::Kz2;
            rec->apply_multiplier(const_cast<double*>(d_q2_phys), args.d_phys_out, type, coeff, args.stream);
        }

        // ===== Step 3: Reduce physical grid back to reduced basis =====
        const double* d_tmp_reduced;
        if (args.identity_map)
        {
            d_tmp_reduced = args.d_phys_out;
        }
        else
        {
            ker_reduce_to_basis<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
                args.d_reduce_buf, args.d_phys_out, args.d_reduced_to_phys, args.n_basis);
            gpu_error_check(cudaPeekAtLastError());
            d_tmp_reduced = args.d_reduce_buf;
        }

        // ===== Step 4: Orbit-weighted dot product with q1 =====
        ker_multi<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
            args.d_reduce_buf, d_tmp_reduced, d_q1_reduced, 1.0, args.n_basis);
        gpu_error_check(cudaPeekAtLastError());
        ker_multi_weight<<<N_BLOCKS, N_THREADS, 0, args.stream>>>(
            args.d_reduce_buf, args.d_reduce_buf, args.d_orbit_counts, args.n_basis);
        gpu_error_check(cudaPeekAtLastError());

        size_t temp_bytes = args.cub_temp_bytes;
        cub::DeviceReduce::Sum(args.d_cub_temp, temp_bytes,
                               args.d_reduce_buf, &d_segment_stress[d], args.n_basis, args.stream);
        gpu_error_check(cudaPeekAtLastError());
    }

    // ===== Step 5: Symmetrize equivalent axes and scale =====
    // The multipliers above are weighted by Cartesian k_d² = (2πm_d/L_d)²,
    // while the standard path accumulates deformation-vector components
    // v_d² = k_d²/L_d². Divide each sum by L_d² so both paths return the
    // same V_dd, and scale by b²·M_full (FFT normalization convention).
    const std::string& crystal_system = args.space_group->get_crystal_system();
    int sym_mode = 0;
    if (crystal_system == "Cubic")
        sym_mode = 2;
    else if (crystal_system == "Tetragonal")
        sym_mode = 1;

    const double s0 = args.bond_length_sq * args.M_full / (args.lx[0] * args.lx[0]);
    const double s1 = args.bond_length_sq * args.M_full / (args.lx[1] * args.lx[1]);
    const double s2 = args.bond_length_sq * args.M_full / (args.lx[2] * args.lx[2]);

    ker_crysfft_stress_finalize<<<1, 1, 0, args.stream>>>(d_segment_stress, sym_mode, s0, s1, s2);
    gpu_error_check(cudaPeekAtLastError());
}
