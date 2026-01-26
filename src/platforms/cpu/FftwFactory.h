/**
 * @file FftwFactory.h
 * @brief Concrete factory for FFTW3-based CPU implementations.
 *
 * This header provides the FftwFactory class, which is the concrete implementation
 * of AbstractFactory for CPU-based computations using FFTW3 (Fastest Fourier
 * Transform in the West). It creates CPU-optimized versions of all simulation objects.
 *
 * **Platform Features:**
 *
 * - Uses FFTW3 for FFT operations (O(N log N) for all transform types)
 * - OpenMP parallelization for multi-core CPUs
 * - Supports both continuous and discrete chain models
 * - Supports pseudo-spectral and real-space solvers
 * - Native DCT/DST support via FFTW r2r transforms
 *
 * **Performance Notes:**
 *
 * - FFTW provides O(N log N) DCT/DST (faster than the old O(N^2) matrix multiplication)
 * - Best for simulations with non-periodic boundary conditions
 * - Can be previously with slower for certain grid sizes
 *
 * **License Note:**
 *
 * FFTW3 is licensed under GPL. Using this factory requires compliance with GPL terms.
 *
 * @see AbstractFactory for the factory interface
 * @see FftwFactory for FFTW implementation
 * @see PlatformSelector for factory instantiation
 *
 * @example
 * @code
 * // Create FFTW factory
 * FftwFactory<double> factory(false);  // reduce_memory=false for speed
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

#ifndef FFTW_FACTORY_H_
#define FFTW_FACTORY_H_

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "Array.h"

/**
 * @class FftwFactory
 * @brief Factory for creating FFTW3-based CPU simulation objects.
 *
 * Implements AbstractFactory to create CPU-optimized objects using FFTW3.
 * All created objects operate on host memory and use OpenMP for parallelization.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Created Object Types:**
 *
 * - CpuComputationBox: Grid and integration on CPU
 * - CpuComputationContinuous/Discrete: Propagator solvers (using FFTW)
 * - CpuAndersonMixing: Iteration accelerator
 *
 * @note The reduce_memory flag enables checkpointing mode for memory-constrained
 *       simulations at the cost of increased computation time.
 */
template <typename T>
class FftwFactory : public AbstractFactory<T>
{
public :
    /**
     * @brief Construct an FFTW factory.
     * @param reduce_memory Enable checkpointing mode (reduces memory, increases compute)
     */
    FftwFactory(bool reduce_memory);

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
     * based on chain model and numerical method. Uses FFTW3 for FFT operations.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Computation scheduler
     * @param numerical_method                "rqm4", "etdrk4" (pseudo-spectral) or
     *                                        "cn-adi2", "cn-adi4-lr" (real-space)
     * @return CPU propagator solver (using FFTW)
     */
    PropagatorComputation<T>* create_propagator_computation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string numerical_method, SpaceGroup* space_group=nullptr) override;

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
     * Prints FFTW version and other configuration details.
     */
    void display_info() override;
};
#endif
