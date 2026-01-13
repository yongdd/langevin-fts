/**
 * @file MklFactory.h
 * @brief Concrete factory for Intel MKL-based CPU implementations.
 *
 * This header provides the MklFactory class, which is the concrete implementation
 * of AbstractFactory for CPU-based computations using Intel Math Kernel Library (MKL).
 * It creates CPU-optimized versions of all simulation objects.
 *
 * **Platform Features:**
 *
 * - Uses Intel MKL for FFT operations (highly optimized for Intel CPUs)
 * - OpenMP parallelization for multi-core CPUs
 * - Supports both continuous and discrete chain models
 * - Supports pseudo-spectral and real-space solvers
 *
 * **Performance Notes:**
 *
 * - Best for 1D simulations where GPU overhead isn't worthwhile
 * - Good for debugging and validation (easier to debug than GPU code)
 * - Can be faster than GPU for small grids (< 32^3)
 *
 * @see AbstractFactory for the factory interface
 * @see CudaFactory for GPU implementation
 * @see PlatformSelector for factory instantiation
 *
 * @example
 * @code
 * // Create MKL factory
 * MklFactory<double> factory(false);  // reduce_memory_usage ignored for CPU
 *
 * // Create simulation objects
 * auto* cb = factory.create_computation_box(nx, lx, bc);
 * auto* mols = factory.create_molecules_information("continuous", 0.01, bonds);
 * auto* solver = factory.create_propagator_computation(cb, mols, optimizer, "rqm4");
 *
 * // Display platform info
 * factory.display_info();
 * @endcode
 */

#ifndef MKL_FACTORY_H_
#define MKL_FACTORY_H_

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "Array.h"

/**
 * @class MklFactory
 * @brief Factory for creating Intel MKL-based CPU simulation objects.
 *
 * Implements AbstractFactory to create CPU-optimized objects using Intel MKL.
 * All created objects operate on host memory and use OpenMP for parallelization.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Created Object Types:**
 *
 * - CpuComputationBox: Grid and integration on CPU
 * - CpuComputationContinuous/Discrete: Propagator solvers
 * - CpuAndersonMixing: Iteration accelerator
 *
 * @note The reduce_memory_usage flag is accepted for API compatibility
 *       but currently has no effect on CPU implementations.
 */
template <typename T>
class MklFactory : public AbstractFactory<T>
{
public :
    /**
     * @brief Construct an MKL factory.
     * @param reduce_memory_usage Enable memory-saving mode (checkpointing)
     */
    MklFactory(bool reduce_memory_usage);

    /**
     * @brief Create a CPU computation box.
     * @param nx   Grid dimensions
     * @param lx   Box lengths
     * @param bc   Boundary conditions
     * @param mask Optional mask for impenetrable regions
     * @return CpuComputationBox instance
     */
    ComputationBox<T>* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        const double* mask=nullptr) override;

    /**
     * @brief Create a CPU computation box with lattice angles.
     * @param nx     Grid dimensions
     * @param lx     Box lengths
     * @param bc     Boundary conditions
     * @param angles Lattice angles [alpha, beta, gamma] in degrees
     * @param mask   Optional mask for impenetrable regions
     * @return CpuComputationBox instance
     */
    ComputationBox<T>* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        std::vector<double> angles,
        const double* mask=nullptr) override;

    /**
     * @brief Create molecules container.
     * @param chain_model  "continuous" or "discrete"
     * @param ds           Contour step size
     * @param bond_lengths Segment lengths squared
     * @return Molecules instance
     */
    Molecules* create_molecules_information(
        std::string chain_model, double ds, std::map<std::string, double> bond_lengths) override;

    /**
     * @brief Create propagator computation solver.
     *
     * Creates CpuComputationContinuous, CpuComputationDiscrete, or real-space solver
     * based on chain model and numerical method.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Computation scheduler
     * @param numerical_method                "rqm4", "etdrk4" (pseudo-spectral) or
     *                                        "cn-adi2", "cn-adi4" (real-space)
     * @return CPU propagator solver
     */
    PropagatorComputation<T>* create_propagator_computation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string numerical_method) override;

    /**
     * @brief Create CPU Anderson mixing optimizer.
     * @param n_var      Number of field variables
     * @param max_hist   Maximum history length
     * @param start_error Error threshold
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     * @return CpuAndersonMixing instance
     */
    AndersonMixing<T>* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;

    /**
     * @brief Display CPU platform information.
     *
     * Prints MKL version, number of threads, and other configuration details.
     */
    void display_info() override;
};
#endif
