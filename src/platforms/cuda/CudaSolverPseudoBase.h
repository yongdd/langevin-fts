/**
 * @file CudaSolverPseudoBase.h
 * @brief Shared state and operations of the CUDA pseudo-spectral solvers.
 *
 * CudaSolverPseudoRQM4, CudaSolverPseudoRK2 and CudaSolverPseudoDiscrete differ
 * only in how they advance a propagator (and, for the discrete model, in the
 * bond factor used by the stress). Everything else lives here, mirroring
 * CpuSolverPseudoBase on the CPU side:
 *
 * - computation box, Pseudo helper and stream handles,
 * - the spectral transforms: cuFFT plans for periodic boundaries
 *   (fft_forward / fft_backward, batch of 1 or 2 fields) and per-stream
 *   DCT/DST objects (fft_) for non-periodic boundaries,
 * - the stress reduction workspace,
 * - the space-group reduced basis and the CrysFFT fast path.
 *
 * Derived classes own only their propagator work buffers, because the number
 * and size of those buffers is what distinguishes the three methods.
 */

#ifndef CUDA_SOLVER_PSEUDO_BASE_H_
#define CUDA_SOLVER_PSEUDO_BASE_H_

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cufft.h>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "FFT.h"
#include "CudaFFT.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "CudaCrysFFT.h"   // CudaCrysFFTBase, CudaCrysFFTMode

class SpaceGroup;
struct CrysFFTBasisMapping;

template <typename T>
class CudaSolverPseudoBase : public CudaSolver<T>
{
protected:
    ComputationBox<T>* cb;        ///< Computation box
    Molecules *molecules;         ///< Molecules container
    std::unique_ptr<Pseudo<T>> pseudo;   ///< Pseudo-spectral utilities (Fourier basis, Boltzmann bond factors)
    int n_streams;                ///< Number of parallel streams
    int dim_;                     ///< Number of dimensions (1, 2, or 3)
    bool is_periodic_;            ///< True if all BCs are periodic
    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy] per index

    // Spectral transforms (created by init_fft(), released by release_base())
    int max_fft_batch_;                      ///< Largest batch a derived solver transforms at once (1 or 2)
    cufftHandle plan_for_[MAX_STREAMS][2];   ///< Periodic BC: forward plans, index = batch - 1
    cufftHandle plan_bak_[MAX_STREAMS][2];   ///< Periodic BC: backward plans, index = batch - 1
    std::unique_ptr<CudaFFTBase<T>> fft_[MAX_STREAMS];       ///< Non-periodic BC: per-stream DCT/DST transform objects (issued on streams[i][0])

    // Stress reduction workspace (per stream)
    size_t temp_storage_bytes[MAX_STREAMS];
    CudaDeviceBuffer<unsigned char> d_temp_storage[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_stress_sum[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_stress_sum_out[MAX_STREAMS];
    CudaDeviceBuffer<CuDeviceData<T>> d_q_multi[MAX_STREAMS];
    CudaDeviceBuffer<cuDoubleComplex> d_qk_conj_[MAX_STREAMS];   ///< q2(-k) gather of the stress (complex fields; nullptr for real)
    CudaDeviceBuffer<CuDeviceData<T>> d_stress_partial_[MAX_STREAMS];   ///< Per-block partial sums of the stress components

    // Space group symmetry (reduced basis)
    SpaceGroup* space_group_;                        ///< Space group pointer (nullptr if not used)
    int* d_reduced_basis_indices_;                   ///< Device array: reduced -> full index mapping (lent by the backend)
    int* d_full_to_reduced_map_;                     ///< Device array: full -> reduced index mapping (lent by the backend)
    int n_basis_;                                    ///< Number of reduced basis points
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_in_[MAX_STREAMS];      ///< Work buffer: full grid input (per stream)
    CudaDeviceBuffer<CuDeviceData<T>> d_q_full_out_[MAX_STREAMS];     ///< Work buffer: full grid output (per stream)
    int q_full_in_factor_;                           ///< d_q_full_in_ holds this many full-grid fields (2 for discrete stress)

    // CrysFFT fast path (crystallographic FFT on the physical sub-grid)
    bool use_crysfft_;                               ///< True when the CrysFFT path is enabled
    CudaCrysFFTMode crysfft_mode_;
    bool crysfft_identity_map_;                      ///< True when reduced basis matches physical grid
    int crysfft_physical_size_;
    CudaDeviceBuffer<int> d_crysfft_phys_to_reduced_;   ///< Device map: physical index -> reduced index
    CudaDeviceBuffer<int> d_crysfft_reduced_to_phys_;   ///< Device map: reduced index -> physical index
    CudaDeviceBuffer<int> d_crysfft_orbit_counts_;      ///< Device orbit counts (stress fast path)
    std::unique_ptr<CudaCrysFFTBase> crysfft_[MAX_STREAMS];   ///< Per-stream CrysFFT objects
    CudaDeviceBuffer<double> d_crysfft_in_[MAX_STREAMS];              ///< Per-stream physical grid input
    CudaDeviceBuffer<double> d_crysfft_out_[MAX_STREAMS];             ///< Per-stream physical grid output
    std::set<double> crysfft_coeffs_;   ///< Step coefficients whose tables update_crysfft_tables() built

    /**
     * @param max_fft_batch             Largest number of fields the derived solver
     *                                  transforms in one cuFFT call (1 or 2).
     * @param q_full_in_factor          Number of full-grid fields d_q_full_in_ must hold.
     * @param with_exp_dw_half          Also allocate d_exp_dw_half (half-step Boltzmann factors).
     * @param complex_requires_periodic Reject complex fields with non-periodic BCs.
     */
    CudaSolverPseudoBase(
        ComputationBox<T>* cb, Molecules *molecules, int n_streams,
        cudaStream_t streams[MAX_STREAMS][2],
        int max_fft_batch, int q_full_in_factor, bool with_exp_dw_half, bool complex_requires_periodic);

    /// Forward FFT of `batch` consecutive full-grid fields on stream `stream` (periodic BC only).
    void fft_forward(int stream, CuDeviceData<T>* d_in, cuDoubleComplex* d_out, int batch = 1);
    /// Backward (unnormalized) FFT, inverse of fft_forward() (periodic BC only).
    void fft_backward(int stream, cuDoubleComplex* d_in, CuDeviceData<T>* d_out, int batch = 1);

    /// Release the CrysFFT objects and device maps (safe to call repeatedly).
    void cleanup_crysfft();

    /**
     * @brief Full-grid input of a spectral step: d_q_in itself, or its expansion
     *        from the reduced basis into d_q_full_in_ when a space group is set.
     */
    CuDeviceData<T>* expand_to_full_grid(int STREAM, CuDeviceData<T>* d_q_in);

    /// Full-grid output of a spectral step: d_q_out itself, or d_q_full_out_ when a space group is set.
    CuDeviceData<T>* full_grid_output(int STREAM, CuDeviceData<T>* d_q_out);

    /**
     * @brief Store the full-grid result d_full in d_q_out, reducing it to the
     *        basis when a space group is set, then apply the mask (if any).
     */
    void store_full_grid(int STREAM, CuDeviceData<T>* d_full, CuDeviceData<T>* d_q_out, const double* d_q_mask);

    /// d_q *= mask on the propagator basis (reduced basis with a space group); no-op for nullptr.
    void multiply_mask(int STREAM, CuDeviceData<T>* d_q, const double* d_q_mask);

    /**
     * @brief CrysFFT diffusion of reduced-basis propagators on the physical sub-grid.
     *
     * d_q_out = post · D · (mid · D ·) pre · d_q_in with D = exp(coeff ∇²): the
     * factors are reduced-basis Boltzmann factors, nullptr to skip one, and a
     * non-null mid adds the second diffusion step. d_q_out may alias d_q_in.
     */
    void crysfft_diffuse(int STREAM, double* d_q_in, double* d_q_out, double coeff,
                         const double* pre, const double* mid, const double* post);

    /**
     * @brief Shared body of update_dw(): d_exp_dw = exp(-factor*ds*w) and,
     *        if with_half, d_exp_dw_half = exp(-factor_half*ds*w).
     */
    void update_dw_impl(
        const std::string& device, const std::map<std::string, const T*>& w_input,
        bool with_half, double factor, double factor_half);

    /**
     * @brief Device scratch a derived solver lends to compute_single_segment_stress().
     *
     * Buffers of the derived solver that are free while one segment's stress is
     * computed: pair_full takes the two propagators expanded to the full grid
     * (2*M; reduced-basis runs only), qk_pair their batched forward FFT
     * (2*M_COMPLEX), rk_1/rk_2 the real DCT/DST coefficients (M_COMPLEX each,
     * non-periodic BC; nullptr when periodic).
     */
    struct StressScratch
    {
        CuDeviceData<T>* pair_full;
        cuDoubleComplex* qk_pair;
        double* rk_1;
        double* rk_2;
    };
    virtual StressScratch stress_scratch(int STREAM) = 0;

    /**
     * @brief Boltzmann bond factor multiplied into the stress integrand.
     *
     * The default (continuous chains) returns nullptr; the discrete chain
     * overrides it with exp(-k^2 b^2 ds/6) (half-bond variant when requested).
     */
    virtual const double* get_stress_boltz_bond(
        const std::string& monomer_type, bool is_half_bond_length) const;

    /**
     * @brief CrysFFT diffusion coefficient b²·Δs/divisor of one propagator step.
     *
     * @param divisor 6 for a full step, 12 for a half step
     */
    double crysfft_coeff(const std::string& monomer_type, int ds_index, double divisor) const;

    /**
     * @brief Coefficients (see crysfft_coeff()) of the CrysFFT steps that the
     *        method takes for a block of this monomer type and contour step.
     *
     * Their Boltzmann tables are built for every block when the CrysFFT objects
     * are created and when the cell changes, so the propagator loop only selects
     * them; crysfft_diffuse() rejects a coefficient whose table was not built.
     */
    virtual std::vector<double> crysfft_block_coeffs(const std::string& monomer_type, int ds_index) const = 0;

private:
    bool plan_for_created_[MAX_STREAMS][2];  ///< plan_for_[i][b] exists (cufftHandle has no null value)
    bool plan_bak_created_[MAX_STREAMS][2];  ///< plan_bak_[i][b] exists

    /// Free everything the base owns (idempotent; safe on a partly built object).
    void release_base();
    /// Create the cuFFT plans (periodic BC) or the DCT/DST objects (non-periodic BC).
    void init_fft();
    /// Select and build the CrysFFT fast path for the current space group (3D periodic, real fields).
    void init_crysfft();
    /// Upload the index maps and create the per-stream CrysFFT objects for `mode`.
    void setup_crysfft(CudaCrysFFTMode mode, const CrysFFTBasisMapping& map,
                       const std::array<double, 9>& m3_translations);
    /// Pass the current cell to the CrysFFT objects and build the Boltzmann tables of all steps.
    void update_crysfft_tables();

public:
    virtual ~CudaSolverPseudoBase();

    /**
     * @brief Stress contribution of one segment: V_ij = sum_k q1(k) q2(-k) v_i v_j.
     *
     * Shared by RQM4, RK2 and the discrete chain; they differ only in the
     * Boltzmann bond factor (get_stress_boltz_bond()) and in the scratch
     * buffers they lend (stress_scratch()). Paths: CrysFFT fast path for
     * orthogonal 3D reduced-basis runs, batched cuFFT for periodic BC, DCT/DST
     * for non-periodic BC (real fields). Cross terms are accumulated only for
     * non-orthogonal boxes or when force_off_diagonal_stress_ is set.
     *
     * @param STREAM           Stream index for concurrent execution
     * @param d_q_pair         Forward and backward propagators (device, contiguous,
     *                         the second at offset n_basis_ / total_grid)
     * @param d_segment_stress Output: [V11, V22, V33, V12, V13, V23] (device)
     * @param monomer_type     Monomer type (bond length)
     * @param is_half_bond_length Half-bond contribution (discrete chain)
     */
    void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;

    void set_space_group(
        SpaceGroup* sg,
        int* d_reduced_basis_indices,
        int* d_full_to_reduced_map,
        int n_basis) override;

    void update_laplacian_operator() override;
};

#endif  // CUDA_SOLVER_PSEUDO_BASE_H_
