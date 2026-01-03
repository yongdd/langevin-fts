/**
 * @file CudaSolverPseudoMixedBC.h
 * @brief GPU pseudo-spectral solver with mixed boundary conditions.
 *
 * This header provides CudaSolverPseudoMixedBC, the CUDA implementation
 * of the pseudo-spectral method supporting mixed boundary conditions
 * (reflecting/absorbing) using DCT/DST transforms.
 *
 * **GPU Architecture:**
 *
 * - Multiple CUDA streams for concurrent propagator computation
 * - Custom DCT/DST kernels (cuFFT for periodic if all dimensions periodic)
 * - Custom CUDA kernels for element-wise operations
 *
 * **Boundary Conditions:**
 *
 * - REFLECTING (Neumann): Uses DCT-II/III transforms
 * - ABSORBING (Dirichlet): Uses DST-II/III transforms
 * - Mixed combinations supported dimension-by-dimension
 *
 * @see CpuSolverPseudoMixedBC for CPU version
 * @see CudaSolverPseudoContinuous for standard periodic version
 */

#ifndef CUDA_SOLVER_PSEUDO_MIXED_BC_H_
#define CUDA_SOLVER_PSEUDO_MIXED_BC_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "CudaFFTMixedBC.h"
#include "CudaPseudo.h"

/**
 * @class CudaSolverPseudoMixedBC
 * @brief GPU pseudo-spectral solver for mixed boundary conditions.
 *
 * Implements operator splitting with Richardson extrapolation on GPU
 * using DCT/DST transforms for non-periodic boundaries.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Richardson Extrapolation:**
 *
 * Same algorithm as CPU version:
 *     q(s+ds) = (4/3) q^(ds/2,ds/2) - (1/3) q^(ds)
 *
 * **Memory per Stream:**
 *
 * - d_q_step_*: Workspace for full/half step computations
 * - d_qk_*: Fourier/DCT/DST coefficient buffers
 */
template <typename T>
class CudaSolverPseudoMixedBC : public CudaSolver<T>
{
private:
    ComputationBox<T>* cb;             ///< Computation box
    Molecules* molecules;               ///< Molecules container
    CudaPseudo<T>* pseudo;              ///< Pseudo-spectral utilities (GPU)
    std::string chain_model;            ///< Chain model ("continuous")

    int n_streams;                      ///< Number of parallel streams
    bool is_periodic_;                  ///< True if all BCs are periodic

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /// @name cuFFT Resources (for periodic BC fallback)
    /// @{
    cufftHandle plan_for_one[MAX_STREAMS];   ///< Single forward FFT plans
    cufftHandle plan_bak_one[MAX_STREAMS];   ///< Single backward FFT plans
    /// @}

    /// @name DCT/DST Transform Objects (per dimension)
    /// @{
    CudaFFTMixedBC<T, 1>* fft_mixed_1d;   ///< 1D DCT/DST transform
    CudaFFTMixedBC<T, 2>* fft_mixed_2d;   ///< 2D DCT/DST transform
    CudaFFTMixedBC<T, 3>* fft_mixed_3d;   ///< 3D DCT/DST transform
    /// @}

    /// @name Workspace Arrays (per stream)
    /// @{
    CuDeviceData<T>* d_q_step_1_one[MAX_STREAMS];   ///< Full step workspace 1
    CuDeviceData<T>* d_q_step_2_one[MAX_STREAMS];   ///< Full step workspace 2
    CuDeviceData<T>* d_q_step_1_two[MAX_STREAMS];   ///< Half step workspace (2 fields)
    /// @}

    /// @name Fourier/DCT Space Buffers (per stream)
    /// @{
    double* d_qk_in_1[MAX_STREAMS];      ///< DCT/DST coefficient buffer 1
    double* d_qk_in_2[MAX_STREAMS];      ///< DCT/DST coefficient buffer 2
    cuDoubleComplex* d_qk_complex_1[MAX_STREAMS];  ///< Complex buffer for periodic
    cuDoubleComplex* d_qk_complex_2[MAX_STREAMS];  ///< Complex buffer for periodic
    /// @}

    /// @name Stress Computation Buffers (always double for DCT/DST)
    /// @{
    size_t temp_storage_bytes[MAX_STREAMS];
    double* d_temp_storage[MAX_STREAMS];
    double* d_stress_sum[MAX_STREAMS];
    double* d_q_multi[MAX_STREAMS];
    /// @}

    /**
     * @brief Transform forward (dispatch to DCT/DST or cuFFT).
     */
    void transform_forward(const int STREAM, CuDeviceData<T>* d_rdata, double* d_cdata);

    /**
     * @brief Transform backward (dispatch to DCT/DST or cuFFT).
     */
    void transform_backward(const int STREAM, double* d_cdata, CuDeviceData<T>* d_rdata);

public:
    /**
     * @brief Construct GPU pseudo-spectral solver for mixed BCs.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory_usage Memory saving mode
     */
    CudaSolverPseudoMixedBC(
        ComputationBox<T>* cb,
        Molecules* molecules,
        int n_streams,
        cudaStream_t streams[MAX_STREAMS][2],
        bool reduce_memory_usage);

    /**
     * @brief Destructor. Frees GPU memory and cuFFT plans.
     */
    ~CudaSolverPseudoMixedBC();

    /** @brief Update Fourier-space operators for new box dimensions. */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors.
     * @param device  "gpu" or "cpu" for w_input location
     * @param w_input Potential fields
     */
    void update_dw(std::string device, std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one step using Richardson extrapolation.
     *
     * @param STREAM      Stream index for concurrent execution
     * @param d_q_in      Input propagator (device)
     * @param d_q_out     Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask (device, nullptr if none)
     */
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out,
        std::string monomer_type, double* d_q_mask) override;

    /** @brief Half-bond step (empty for continuous chains). */
    void advance_propagator_half_bond_step(
        const int, CuDeviceData<T>*, CuDeviceData<T>*, std::string) override {};

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
        CuDeviceData<T>* d_q_pair, CuDeviceData<T>* d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};

#endif
