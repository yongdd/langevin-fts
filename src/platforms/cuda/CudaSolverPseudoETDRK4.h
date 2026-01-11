/**
 * @file CudaSolverPseudoETDRK4.h
 * @brief GPU ETDRK4 pseudo-spectral solver for continuous chain model.
 *
 * This header provides CudaSolverPseudoETDRK4, the CUDA implementation
 * of the ETDRK4 (Exponential Time Differencing Runge-Kutta 4th order)
 * method for continuous Gaussian chains.
 *
 * **GPU Architecture:**
 *
 * - Multiple CUDA streams for concurrent propagator computation
 * - cuFFT for GPU-accelerated FFT operations
 * - Custom CUDA kernels for element-wise operations
 *
 * **ETDRK4 Algorithm (Cox & Matthews 2002):**
 *
 * For dq/ds = L·q + N(q) where L = (b²/6)∇² and N(q) = -w·q:
 * - Stage a: â = E2·q̂ + α·N̂_n
 * - Stage b: b̂ = E2·q̂ + α·N̂_a
 * - Stage c: ĉ = E2·â + α·(2N̂_b - N̂_n)
 * - Final: q̂_{n+1} = E·q̂ + f1·N̂_n + f2·(N̂_a + N̂_b) + f3·N̂_c
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
 * Implements the ETDRK4 method with Kassam-Trefethen coefficient computation
 * for stable and L-stable 4th-order time stepping.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **ETDRK4 vs RQM4:**
 *
 * - ETDRK4: L-stable, may allow larger step sizes
 * - RQM4: Simpler, well-tested
 *
 * Both methods are 4th-order accurate.
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

    /// @name ETDRK4 Coefficients
    /// @{
    std::unique_ptr<ETDRK4Coefficients<T>> etdrk4_coefficients_;  ///< ETDRK4 coefficient arrays (host)
    std::map<std::string, double*> d_etdrk4_E;     ///< exp(c*h) on device
    std::map<std::string, double*> d_etdrk4_E2;    ///< exp(c*h/2) on device
    std::map<std::string, double*> d_etdrk4_alpha; ///< h*phi_1(c*h/2) on device
    std::map<std::string, double*> d_etdrk4_f1;    ///< h*beta_1(c*h) on device
    std::map<std::string, double*> d_etdrk4_f2;    ///< h*beta_2(c*h) on device
    std::map<std::string, double*> d_etdrk4_f3;    ///< h*beta_3(c*h) on device
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
     * @param reduce_memory_usage Memory saving mode (affects workspace allocation)
     */
    CudaSolverPseudoETDRK4(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory_usage);

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
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask) override;

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
