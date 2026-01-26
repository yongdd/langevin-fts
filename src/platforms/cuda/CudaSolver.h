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
 * @see CudaSolverPseudoRQM4 for continuous chain pseudo-spectral
 * @see CudaSolverPseudoDiscrete for discrete chain pseudo-spectral
 * @see CudaSolverCNADI for finite difference method
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

class SpaceGroup;  // Forward declaration

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
 * - d_exp_dw: Full segment Boltzmann factor (per ds_index for local ds support)
 * - d_exp_dw_half: Half segment Boltzmann factor (per ds_index)
 */
template <typename T>
class CudaSolver
{
public:
    /**
     * @brief Full segment Boltzmann factor: exp(-w·ds) on GPU.
     *
     * Outer key: ds_index (1-based)
     * Inner key: monomer type
     * Value: Device pointer to array (size n_grid)
     */
    std::map<int, std::map<std::string, CuDeviceData<T>*>> d_exp_dw;

    /**
     * @brief Half segment Boltzmann factor: exp(-w·ds/2) on GPU.
     *
     * Outer key: ds_index (1-based)
     * Inner key: monomer type
     * Value: Device pointer to array (size n_grid)
     */
    std::map<int, std::map<std::string, CuDeviceData<T>*>> d_exp_dw_half;

    /**
     * @brief Virtual destructor.
     */
    virtual ~CudaSolver() {};

    /**
     * @brief Set space group for reduced basis expand/reduce operations.
     *
     * When set, the solver will internally handle expand (reduced→full) before
     * FFT and reduce (full→reduced) after FFT. Input/output are in reduced basis.
     *
     * @param sg                      SpaceGroup pointer (nullptr to disable)
     * @param d_reduced_basis_indices Device array mapping reduced→full indices
     * @param d_full_to_reduced_map   Device array mapping full→reduced indices
     * @param n_basis                 Number of reduced basis points
     */
    virtual void set_space_group(
        [[maybe_unused]] SpaceGroup* sg,
        [[maybe_unused]] int* d_reduced_basis_indices,
        [[maybe_unused]] int* d_full_to_reduced_map,
        [[maybe_unused]] int n_basis) {}

    /**
     * @brief Reset internal propagator states for a stream.
     *
     * Used by Global Richardson method to reset per-stream state.
     * Default implementation does nothing (for solvers without internal state).
     *
     * @param STREAM Stream index to reset
     */
    virtual void reset_internal_state([[maybe_unused]] int STREAM) {}

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
     * @param ds_index    Index for ds value (1-based, for per-block ds support)
     */
    virtual void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) = 0;

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

    /**
     * @brief Apply mask to propagator.
     *
     * When space_group is set, expands propagator to full grid, multiplies
     * by mask, and reduces back to reduced basis. Otherwise multiplies directly.
     *
     * @param STREAM  CUDA stream index
     * @param d_q     Propagator array (device, modified in place)
     * @param d_mask  Mask array (device, always full grid)
     */
    virtual void apply_mask(
        [[maybe_unused]] const int STREAM,
        [[maybe_unused]] CuDeviceData<T> *d_q,
        [[maybe_unused]] double *d_mask) {}
};
#endif
