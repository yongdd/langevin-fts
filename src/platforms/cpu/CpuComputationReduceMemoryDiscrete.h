/**
 * @file CpuComputationReduceMemoryDiscrete.h
 * @brief Memory-efficient CPU propagator computation for discrete chains.
 *
 * This header provides CpuComputationReduceMemoryDiscrete, a variant of
 * CpuComputationDiscrete that trades computation time for reduced memory
 * usage through checkpointing.
 *
 * **Checkpointing Strategy:**
 *
 * Instead of storing propagators at every segment, this implementation:
 *
 * 1. Stores propagators only at checkpoint intervals (dependency points)
 * 2. Recomputes intermediate steps on-the-fly during concentration calculation
 *
 * **Memory vs Time Tradeoff:**
 *
 * - Standard: O(N × M) memory, O(N × M) computation
 * - Checkpointing: O(√N × M) memory, O(N × √N × M) computation
 *
 * where N is segment count and M is grid points.
 *
 * **Discrete Chain Specifics:**
 *
 * Unlike continuous chains, discrete chains have:
 * - Full segments at regular positions
 * - Half-bond steps at junction points
 * - Different concentration calculation (no Simpson's rule)
 *
 * **When to Use:**
 *
 * - Large 3D grids where propagator storage exceeds available memory
 * - Long polymer chains (large N)
 * - Systems with many polymer species
 *
 * @see CpuComputationDiscrete for standard discrete version
 * @see CpuComputationReduceMemoryContinuous for continuous memory-efficient version
 * @see PropagatorComputation for the abstract interface
 */

#ifndef CPU_PSEUDO_REDUCE_MEMORY_DISCRETE_H_
#define CPU_PSEUDO_REDUCE_MEMORY_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "CpuComputationReduceMemoryBase.h"
#include "CpuSolver.h"
#include "Scheduler.h"
#include "FFT.h"  // For FFTBackend enum

/**
 * @class CpuComputationReduceMemoryDiscrete
 * @brief Memory-efficient propagator computation for discrete chains using checkpointing.
 *
 * Reduces memory usage by storing only checkpoint propagators and
 * recomputing intermediate values during concentration calculation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Checkpoint Placement:**
 *
 * Checkpoints are placed at dependency points (junctions, chain ends)
 * to minimize storage while enabling recomputation.
 *
 * **Recomputation Process:**
 *
 * During concentration calculation:
 * 1. Load propagator from nearest checkpoint
 * 2. Recompute forward to required segment position
 * 3. Use for concentration contribution
 * 4. Repeat for next position
 *
 * **Half-Bond Steps:**
 *
 * For discrete chains, half-bond propagators at junctions are also
 * stored at checkpoints for proper junction handling.
 *
 * @note This version is slower but uses significantly less memory.
 *       Use for large-scale 3D simulations with discrete chains.
 *
 * @example
 * @code
 * // Enable via reduce_memory parameter in factory
 * FftwFactory<double> factory(true);  // reduce_memory = true
 *
 * // Usage is identical to standard version
 * auto* comp = factory.create_discrete_solver(cb, molecules, optimizer);
 * comp->compute_statistics(w_fields);
 * @endcode
 */
template <typename T>
class CpuComputationReduceMemoryDiscrete : public CpuComputationReduceMemoryBase<T>
{
private:
    /**
     * @brief Half-bond step propagators at checkpoints.
     *
     * Key: (propagator_key, segment_index)
     * Value: Half-bond propagator at junction (size n_grid)
     *
     * Stores half-bond propagators at junction points.
     */
    std::map<std::tuple<std::string, int>, T *> propagator_half_steps_at_check_point;

    #ifndef NDEBUG
    /**
     * @brief Debug: track computed half-step propagators.
     */
    std::map<std::string, std::map<int, bool>> propagator_half_steps_finished;
    #endif

    /**
     * @brief Segment pairs for partition function.
     *
     * Tuple: (polymer_id, q_forward_ptr, q_backward_ptr, monomer_type, n_repeated)
     * Includes monomer_type for proper Boltzmann factor weighting.
     */
    std::vector<std::tuple<int, T *, T *, std::string, int>> single_partition_segment;

    /**
     * @brief Calculate concentration for one block with recomputation.
     *
     * Recomputes propagators from checkpoints as needed.
     * Uses discrete sum instead of integral.
     *
     * @param phi          Output concentration
     * @param key_left     Left propagator key
     * @param key_right    Right propagator key
     * @param N_LEFT       Left contour index
     * @param N_RIGHT      Right contour index
     * @param monomer_type Monomer type
     */
    void calculate_phi_one_block(T *phi, std::string key_left, std::string key_right, const int N_LEFT, const int N_RIGHT, std::string monomer_type);

    /**
     * @brief Recompute propagator from checkpoint.
     *
     * Starting from nearest checkpoint, advances propagator to
     * the required segment positions.
     *
     * @param key          Propagator key
     * @param N_START      Starting segment (checkpoint position)
     * @param N_RIGHT      Ending segment position
     * @param monomer_type Monomer type
     *
     * @return Vector of propagator pointers for requested range
     */
    std::vector<T*> recalculate_propagator(std::string key, const int N_START, const int N_RIGHT, std::string monomer_type);

public:
    /**
     * @brief Construct memory-efficient computation for discrete chains.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer
     * @param backend                         FFT backend to use (FFTW, default: FFTW)
     */
    CpuComputationReduceMemoryDiscrete(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, FFTBackend backend = FFTBackend::FFTW);

    /**
     * @brief Destructor. Frees checkpoints and workspace.
     */
    ~CpuComputationReduceMemoryDiscrete();

    /**
     * @brief Compute propagators and store checkpoints.
     *
     * Only checkpoint values are stored, not full propagator history.
     *
     * @param w_block Potential fields
     * @param q_init  Optional initial conditions
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Calculate concentrations with on-the-fly recomputation.
     *
     * Recomputes propagator values from checkpoints as needed.
     */
    void compute_concentrations() override;

    /**
     * @brief Compute stress (requires recomputation).
     */
    void compute_stress() override;

    /**
     * @brief Get partition function.
     */
    T get_total_partition(int polymer) override;

    /**
     * @brief Get propagator (requires recomputation).
     */
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    /**
     * @brief Validate partition function.
     */
    bool check_total_partition() override;

};
#endif
