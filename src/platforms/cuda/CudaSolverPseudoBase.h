/**
 * @file CudaSolverPseudoBase.h
 * @brief Shared state and operations of the CUDA pseudo-spectral solvers.
 *
 * CudaSolverPseudoRQM4, CudaSolverPseudoRK2 and CudaSolverPseudoDiscrete differ
 * only in how they advance a propagator (and, for the discrete model, in the
 * bond factor used by the stress). Everything else -- computation box, the
 * Pseudo helper, stream handles, the stress reduction workspace, the space
 * group reduced basis and the CrysFFT fast path -- lives here, mirroring
 * CpuSolverPseudoBase on the CPU side.
 *
 * Derived classes own their cuFFT plans, spectral-transform objects and
 * propagator work buffers, because the number and size of those buffers is
 * what distinguishes the three methods.
 */

#ifndef CUDA_SOLVER_PSEUDO_BASE_H_
#define CUDA_SOLVER_PSEUDO_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <cufft.h>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "CudaCrysFFT.h"   // CudaCrysFFTBase, CudaCrysFFTMode

class SpaceGroup;

template <typename T>
class CudaSolverPseudoBase : public CudaSolver<T>
{
protected:
    ComputationBox<T>* cb;        ///< Computation box
    Molecules *molecules;         ///< Molecules container
    Pseudo<T> *pseudo;            ///< Pseudo-spectral utilities (Fourier basis, Boltzmann bond factors)
    std::string chain_model;      ///< "continuous" or "discrete"
    int n_streams;                ///< Number of parallel streams
    int dim_;                     ///< Number of dimensions (1, 2, or 3)
    bool is_periodic_;            ///< True if all BCs are periodic
    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy] per index

    // Stress reduction workspace (per stream)
    size_t temp_storage_bytes[MAX_STREAMS];
    CuDeviceData<T> *d_temp_storage[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum_out[MAX_STREAMS];
    CuDeviceData<T> *d_q_multi[MAX_STREAMS];

    // Space group symmetry (reduced basis)
    SpaceGroup* space_group_;                        ///< Space group pointer (nullptr if not used)
    int* d_reduced_basis_indices_;                   ///< Device array: reduced -> full index mapping
    int* d_full_to_reduced_map_;                     ///< Device array: full -> reduced index mapping
    int n_basis_;                                    ///< Number of reduced basis points
    CuDeviceData<T> *d_q_full_in_[MAX_STREAMS];      ///< Work buffer: full grid input (per stream)
    CuDeviceData<T> *d_q_full_out_[MAX_STREAMS];     ///< Work buffer: full grid output (per stream)
    int q_full_in_factor_;                           ///< d_q_full_in_ holds this many full-grid fields (2 for discrete stress)

    // CrysFFT fast path (crystallographic FFT on the physical sub-grid)
    bool use_crysfft_;                               ///< True when the CrysFFT path is enabled
    CudaCrysFFTMode crysfft_mode_;
    bool crysfft_identity_map_;                      ///< True when reduced basis matches physical grid
    int crysfft_physical_size_;
    int crysfft_reduced_size_;
    int* d_crysfft_phys_to_reduced_;                 ///< Device map: physical index -> reduced index
    int* d_crysfft_reduced_to_phys_;                 ///< Device map: reduced index -> physical index
    int* d_crysfft_orbit_counts_;                    ///< Device orbit counts (stress fast path)
    CudaCrysFFTBase* crysfft_[MAX_STREAMS];          ///< Per-stream CrysFFT objects
    double* d_crysfft_in_[MAX_STREAMS];              ///< Per-stream physical grid input
    double* d_crysfft_out_[MAX_STREAMS];             ///< Per-stream physical grid output

    /**
     * @param q_full_in_factor          Number of full-grid fields d_q_full_in_ must hold.
     * @param with_exp_dw_half          Also allocate d_exp_dw_half (half-step Boltzmann factors).
     * @param complex_requires_periodic Reject complex fields with non-periodic BCs.
     */
    CudaSolverPseudoBase(
        ComputationBox<T>* cb, Molecules *molecules, int n_streams,
        cudaStream_t streams[MAX_STREAMS][2],
        int q_full_in_factor, bool with_exp_dw_half, bool complex_requires_periodic);

    /// Release the CrysFFT objects and device maps (safe to call repeatedly).
    void cleanup_crysfft();

    /**
     * @brief Shared body of update_dw(): d_exp_dw = exp(-factor*ds*w) and,
     *        if with_half, d_exp_dw_half = exp(-factor_half*ds*w).
     */
    void update_dw_impl(
        const std::string& device, const std::map<std::string, const T*>& w_input,
        bool with_half, double factor, double factor_half);

public:
    virtual ~CudaSolverPseudoBase();

    void set_space_group(
        SpaceGroup* sg,
        int* d_reduced_basis_indices,
        int* d_full_to_reduced_map,
        int n_basis) override;

    void update_laplacian_operator() override;
};

#endif  // CUDA_SOLVER_PSEUDO_BASE_H_
