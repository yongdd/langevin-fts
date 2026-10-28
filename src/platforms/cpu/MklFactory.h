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
 * MklFactory<double> factory(false);  // reduce_memory=false for speed
 *
 * // Create simulation objects
 * auto cb = factory.create_computation_box(nx, lx, bc);
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
#include "SpaceGroup.h"

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
 * - Propagator computations with the CPU backend (MKL transforms)
 * - CpuAndersonMixing: Iteration accelerator
 *
 * @note The reduce_memory flag enables checkpointing mode for memory-constrained
 *       simulations at the cost of increased computation time.
 */
template <typename T>
class MklFactory : public AbstractFactory<T>
{
public :
    /**
     * @brief Construct an MKL factory.
     * @param reduce_memory Enable checkpointing mode (reduces memory, increases compute)
     */
    MklFactory(bool reduce_memory);

    /**
     * @brief Create a CPU computation box.
     * @param nx   Grid dimensions
     * @param lx   Box lengths
     * @param bc   Boundary conditions
     * @param mask Optional mask for impenetrable regions
     * @return CpuComputationBox instance
     */
    std::unique_ptr<ComputationBox<T>> create_computation_box(
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
    std::unique_ptr<ComputationBox<T>> create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        std::vector<double> angles,
        const double* mask=nullptr) override;

    /**
     * @brief Create CPU Anderson mixing optimizer.
     * @param n_var      Number of field variables
     * @param max_hist   Maximum history length
     * @param start_error Error threshold
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     * @return CpuAndersonMixing instance
     */
    std::unique_ptr<AndersonMixing<T>> create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;

    /**
     * @brief Create a CPU (MKL) Fourier transform.
     * @param cb Computation box
     * @param bc Transform of each axis
     * @return CpuFourierTransform instance
     */
    std::unique_ptr<FourierTransform<T>> create_fourier_transform(
        ComputationBox<T>* cb, std::vector<BoundaryCondition> bc) override;

    /**
     * @brief Display CPU platform information.
     *
     * Prints MKL version, number of threads, and other configuration details.
     */
    void display_info() override;

protected:
    /// CPU backend with MKL transforms (see AbstractFactory::create_backend())
    std::unique_ptr<ComputationBackend<T>> create_backend(
        ComputationBox<T>* cb, Molecules* molecules, const NumericalMethodInfo& method_info,
        SpaceGroup* space_group, int max_n_streams) override;
};
#endif
