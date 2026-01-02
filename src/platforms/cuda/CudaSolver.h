/**
 * @file CudaSolver.h
 * @brief Abstract interface for GPU propagator solvers.
 *
 * This header defines CudaSolver, the base class for GPU-based propagator
 * solvers. It provides a common interface for pseudo-spectral and real-space
 * methods implemented using CUDA.
 *
 * **Multi-Stream Architecture:**
 *
 * All solver methods take a STREAM parameter for concurrent execution:
 * - Up to MAX_STREAMS propagators can be computed simultaneously
 * - Each stream has independent cuFFT plans and workspace
 *
 * **Device Memory:**
 *
 * All input/output arrays are device pointers (CuDeviceData<T>).
 * No host-device transfers occur within solver methods.
 *
 * @see CudaSolverPseudoContinuous for continuous chain pseudo-spectral
 * @see CudaSolverPseudoDiscrete for discrete chain pseudo-spectral
 * @see CudaSolverRealSpace for finite difference method
 */

#ifndef CUDA_SOLVER_H_
#define CUDA_SOLVER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CudaCommon.h"

/**
 * @class CudaSolver
 * @brief Abstract base class for GPU propagator solvers.
 *
 * Defines the interface for GPU-accelerated propagator advancement
 * and stress computation. Uses CUDA streams for concurrent execution.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Boltzmann Factors:**
 *
 * Like CpuSolver, stores precomputed exp(-w·ds) factors in GPU memory:
 * - d_exp_dw: Full segment Boltzmann factor
 * - d_exp_dw_half: Half segment Boltzmann factor
 */
template <typename T>
class CudaSolver
{
public:
    /**
     * @brief Full segment Boltzmann factor: exp(-w·ds) on GPU.
     *
     * Key: monomer type
     * Value: Device pointer to array (size n_grid)
     */
    std::map<std::string, CuDeviceData<T>*> d_exp_dw;

    /**
     * @brief Half segment Boltzmann factor: exp(-w·ds/2) on GPU.
     *
     * Key: monomer type
     * Value: Device pointer to array (size n_grid)
     */
    std::map<std::string, CuDeviceData<T>*> d_exp_dw_half;

    /**
     * @brief Virtual destructor.
     */
    virtual ~CudaSolver() {};

    /**
     * @brief Update Laplacian operator for new box dimensions.
     *
     * Recomputes Fourier-space operators or finite difference coefficients.
     */
    virtual void update_laplacian_operator() = 0;

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * @param device  Memory location of w_input ("device" or "host")
     * @param w_input Map of potential fields by monomer type
     */
    virtual void update_dw(std::string device, std::map<std::string, const T*> w_input) = 0;

    /**
     * @brief Advance propagator by one contour step on GPU.
     *
     * @param STREAM      CUDA stream index (0 to MAX_STREAMS-1)
     * @param d_q_in      Input propagator (device pointer)
     * @param d_q_out     Output propagator (device pointer)
     * @param monomer_type Monomer type for Boltzmann factor
     * @param d_q_mask    Optional mask for impenetrable regions (device)
     */
    virtual void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask) = 0;

    /**
     * @brief Advance propagator by half bond step (discrete chains).
     *
     * @param STREAM      CUDA stream index
     * @param q_in        Input propagator (device)
     * @param q_out       Output propagator (device)
     * @param monomer_type Monomer type
     */
    virtual void advance_propagator_half_bond_step(
        const int STREAM,
        CuDeviceData<T> *q_in, CuDeviceData<T> *q_out, std::string monomer_type) = 0;

    /**
     * @brief Compute stress contribution from one segment.
     *
     * @param STREAM           CUDA stream index
     * @param d_q_pair         Forward × backward propagator product (device)
     * @param d_segment_stress Output stress array (device)
     * @param monomer_type     Monomer type
     * @param is_half_bond_length Whether using half bond (discrete)
     */
    virtual void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) = 0;
};
#endif
