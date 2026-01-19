/**
 * @file CudaSolverPseudoETDRK4.h
 * @brief GPU ETDRK4 pseudo-spectral solver for continuous chain model.
 *
 * This header provides CudaSolverPseudoETDRK4, the CUDA implementation
 * of the ETDRK4 (Exponential Time Differencing Runge-Kutta 4th order)
 * method for continuous Gaussian chains using the Krogstad scheme.
 *
 * **GPU Architecture:**
 *
 * - Multiple CUDA streams for concurrent propagator computation
 * - cuFFT for GPU-accelerated FFT operations
 * - Custom CUDA kernels for element-wise operations
 *
 * **Krogstad ETDRK4 Algorithm (Song et al. 2018, Eq. 7a-7d):**
 *
 * For dq/ds = L·q + N(q) where L = (b²/6)∇² and N(q) = -w·q:
 * - Stage a (7a): â = E2·q̂ + α·N̂_n
 * - Stage b (7b): b̂ = â + φ₂_half·(N̂_a - N̂_n)
 * - Stage c (7c): ĉ = E·q̂ + φ₁·N̂_n + 2φ₂·(N̂_b - N̂_n)
 * - Final (7d):   q̂_{n+1} = ĉ + (4φ₃ - φ₂)·(N̂_n + N̂_c)
 *                 + 2φ₂·N̂_a - 4φ₃·(N̂_a + N̂_b)
 *
 * Coefficients computed using Kassam-Trefethen (2005) contour integral.
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverPseudoRQM4 for RQM4 version
 * @see CpuSolverPseudoETDRK4 for CPU version
 */

#ifndef CUDA_SOLVER_PSEUDO_ETDRK4_H_
#define CUDA_SOLVER_PSEUDO_ETDRK4_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "CudaFFT.h"
#include "FFT.h"
#include "ETDRK4Coefficients.h"

/**
 * @class CudaSolverPseudoETDRK4
 * @brief GPU ETDRK4 pseudo-spectral solver for continuous Gaussian chains.
 *
 * Implements the Krogstad ETDRK4 scheme (Song et al. 2018) with
 * Kassam-Trefethen coefficient computation for stable time stepping.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 */
template <typename T>
class CudaSolverPseudoETDRK4 : public CudaSolver<T>
{
private:
    ComputationBox<T>* cb;        ///< Computation box
    Molecules *molecules;          ///< Molecules container
    Pseudo<T> *pseudo;             ///< Pseudo-spectral utilities
    std::string chain_model;       ///< Chain model ("continuous")

    int n_streams;                 ///< Number of parallel streams
    int dim_;                      ///< Number of dimensions (1, 2, or 3)
    bool is_periodic_;             ///< True if all BCs are periodic

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy] per index

    /// @name Spectral Transform Resources
    /// @{
    FFT<T>* fft_[MAX_STREAMS];                         ///< Spectral transform objects (per-stream, for non-periodic BC)
    cufftHandle plan_for_one[MAX_STREAMS];             ///< Single forward FFT plans (periodic BC)
    cufftHandle plan_bak_one[MAX_STREAMS];             ///< Single backward FFT plans (periodic BC)
    /// @}

    /// @name Raw Potential Field
    /// @{
    std::map<std::string, CuDeviceData<T>*> d_w_field;    ///< Raw potential field (per monomer type)
    /// @}

    /// @name ETDRK4 Krogstad Coefficients
    /// @{
    std::map<int, std::unique_ptr<ETDRK4Coefficients<T>>> etdrk4_coefficients_;  ///< ETDRK4 coefficient arrays (host), one per ds_index
    std::map<int, std::map<std::string, double*>> d_etdrk4_E;         ///< exp(c*h) on device, [ds_index][monomer_type]
    std::map<int, std::map<std::string, double*>> d_etdrk4_E2;        ///< exp(c*h/2) on device
    std::map<int, std::map<std::string, double*>> d_etdrk4_alpha;     ///< (h/2)*phi_1(c*h/2) on device - stage a
    std::map<int, std::map<std::string, double*>> d_etdrk4_phi2_half; ///< h*phi_2(c*h/2) on device - stage b
    std::map<int, std::map<std::string, double*>> d_etdrk4_phi1;      ///< h*phi_1(c*h) on device - stage c
    std::map<int, std::map<std::string, double*>> d_etdrk4_phi2;      ///< h*phi_2(c*h) on device - stages c, final
    std::map<int, std::map<std::string, double*>> d_etdrk4_phi3;      ///< h*phi_3(c*h) on device - final step
    /// @}

    /// @name ETDRK4 Workspace Arrays (per stream)
    /// @{
    CuDeviceData<T> *d_etdrk4_a[MAX_STREAMS];      ///< ETDRK4 stage a workspace
    CuDeviceData<T> *d_etdrk4_b[MAX_STREAMS];      ///< ETDRK4 stage b workspace
    CuDeviceData<T> *d_etdrk4_c[MAX_STREAMS];      ///< ETDRK4 stage c workspace
    CuDeviceData<T> *d_etdrk4_N_n[MAX_STREAMS];    ///< ETDRK4 nonlinear term N_n
    CuDeviceData<T> *d_etdrk4_N_a[MAX_STREAMS];    ///< ETDRK4 nonlinear term N_a
    CuDeviceData<T> *d_etdrk4_N_b[MAX_STREAMS];    ///< ETDRK4 nonlinear term N_b
    CuDeviceData<T> *d_etdrk4_N_c[MAX_STREAMS];    ///< ETDRK4 nonlinear term N_c
    cuDoubleComplex *d_k_q[MAX_STREAMS];           ///< Fourier space q
    cuDoubleComplex *d_k_a[MAX_STREAMS];           ///< Fourier space a
    cuDoubleComplex *d_k_N_n[MAX_STREAMS];         ///< Fourier space N_n
    cuDoubleComplex *d_k_N_a[MAX_STREAMS];         ///< Fourier space N_a
    cuDoubleComplex *d_k_N_b[MAX_STREAMS];         ///< Fourier space N_b
    cuDoubleComplex *d_k_N_c[MAX_STREAMS];         ///< Fourier space N_c
    cuDoubleComplex *d_k_work[MAX_STREAMS];        ///< Fourier space work buffer
    double *d_rk_work[MAX_STREAMS];                ///< Real coefficients workspace (non-periodic)
    /// @}

    /// @name CUB Reduction Storage (per stream)
    /// @{
    size_t temp_storage_bytes[MAX_STREAMS];
    CuDeviceData<T> *d_temp_storage[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum_out[MAX_STREAMS];
    CuDeviceData<T> *d_q_multi[MAX_STREAMS];
    cuDoubleComplex *d_qk_stress[MAX_STREAMS];     ///< Fourier coefficients for stress
    /// @}

public:
    /**
     * @brief Construct GPU ETDRK4 solver for continuous chains.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory   Checkpointing mode (affects workspace allocation)
     */
    CudaSolverPseudoETDRK4(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory);

    /**
     * @brief Destructor. Frees GPU memory and cuFFT plans.
     */
    ~CudaSolverPseudoETDRK4();

    /** @brief Update Fourier-space operators for new box dimensions. */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors and store raw w field.
     * @param device  "device" or "host" for w_input location
     * @param w_input Potential fields
     */
    void update_dw(std::string device, std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one step using ETDRK4.
     *
     * Uses 4th-order Exponential Time Differencing Runge-Kutta (ETDRK4)
     * with Kassam-Trefethen coefficient computation for L-stability.
     *
     * @param STREAM      Stream index for concurrent execution
     * @param d_q_in      Input propagator (device)
     * @param d_q_out     Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask (device, nullptr if none)
     * @param ds_index    Index for ds value (ETDRK4 uses ds_index=1)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index = 1) override;

    /** @brief Half-bond step (empty for continuous chains). */
    void advance_propagator_half_bond_step(
        const int, CuDeviceData<T> *, CuDeviceData<T> *, std::string) override {};

    /**
     * @brief Compute stress from one segment.
     *
     * @param STREAM           Stream index
     * @param d_q_pair         Product of forward and backward propagators
     * @param d_segment_stress Output stress contribution
     * @param monomer_type     Monomer type
     * @param is_half_bond_length Ignored for continuous
     */
    void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif
