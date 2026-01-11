/**
 * @file AbstractFactory.h
 * @brief Abstract Factory pattern for platform-independent object creation.
 *
 * This header defines the AbstractFactory class, which implements the Abstract
 * Factory design pattern to create platform-specific objects (CPU/MKL or CUDA)
 * without exposing the concrete implementations to client code.
 *
 * **Design Pattern:**
 *
 * The Abstract Factory pattern provides an interface for creating families of
 * related objects without specifying their concrete classes. This enables:
 * - Clean separation between CPU and CUDA implementations
 * - Easy switching between platforms at runtime
 * - Consistent API regardless of backend
 *
 * **Supported Platforms:**
 *
 * - **cpu-mkl**: Intel MKL-based CPU implementation using OpenMP
 * - **cuda**: NVIDIA CUDA GPU implementation using cuFFT
 *
 * @see MklFactory for CPU implementation
 * @see CudaFactory for GPU implementation
 * @see PlatformSelector for factory creation
 *
 * @example
 * @code
 * // Create factory for the desired platform
 * AbstractFactory<double>* factory = PlatformSelector::create_factory_real(
 *     "cuda", false);  // Use CUDA, no memory reduction
 *
 * // Create simulation objects through factory
 * ComputationBox<double>* cb = factory->create_computation_box(nx, lx, bc);
 * Molecules* molecules = factory->create_molecules_information("continuous", ds, bonds);
 * auto* optimizer = factory->create_propagator_computation_optimizer(molecules, true);
 * auto* solver = factory->create_pseudospectral_solver(cb, molecules, optimizer);
 * auto* am = factory->create_anderson_mixing(n_var, 20, 1e-1, 0.1, 0.1);
 *
 * // Display platform information
 * factory->display_info();
 *
 * // Clean up
 * delete solver;
 * delete optimizer;
 * delete molecules;
 * delete cb;
 * delete factory;
 * @endcode
 */

#ifndef ABSTRACT_FACTORY_H_
#define ABSTRACT_FACTORY_H_

#include <string>
#include <array>
#include <complex>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "Array.h"

/**
 * @class AbstractFactory
 * @brief Abstract factory for creating platform-specific simulation objects.
 *
 * This template class defines the interface for creating all major objects
 * needed for polymer field theory simulations. Concrete implementations
 * (MklFactory, CudaFactory) provide platform-optimized versions of each object.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Object Creation:**
 *
 * The factory creates the following objects:
 * - ComputationBox: Grid and integration methods
 * - Molecules: Polymer/solvent container
 * - PropagatorComputationOptimizer: Computation scheduling
 * - PropagatorComputation: Propagator solver (pseudo-spectral or real-space)
 * - AndersonMixing: SCFT iteration accelerator
 *
 * **Memory Usage:**
 *
 * The reduce_memory_usage flag enables memory-saving mode where only
 * propagator checkpoints are stored. This increases computation time
 * 2-4x but significantly reduces memory usage.
 *
 * @see PlatformSelector::create_factory_real for factory instantiation
 */
template <typename T>
class AbstractFactory
{
protected:
    bool reduce_memory_usage;    ///< Enable memory-saving mode
    std::string pseudo_method;   ///< Pseudo-spectral method: "rqm4" or "etdrk4"
    std::string realspace_method; ///< Real-space method: "cn-adi2" or "cn-adi4"

public :
    /**
     * @brief Virtual destructor.
     */
    virtual ~AbstractFactory() {};

    /**
     * @brief Create a ComputationBox for the target platform.
     *
     * @param nx   Number of grid points in each direction [Nx, Ny, Nz]
     * @param lx   Box lengths in each direction [Lx, Ly, Lz]
     * @param bc   Boundary conditions for each face
     * @param mask Optional mask array for impenetrable regions
     * @return Platform-specific ComputationBox instance
     *
     * @example
     * @code
     * std::vector<int> nx = {32, 32, 32};
     * std::vector<double> lx = {4.0, 4.0, 4.0};
     * std::vector<std::string> bc(6, "periodic");
     * auto* cb = factory->create_computation_box(nx, lx, bc);
     * @endcode
     */
    virtual ComputationBox<T>* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        const double* mask=nullptr) = 0;

    /**
     * @brief Create ComputationBox with lattice angles for non-orthogonal systems.
     *
     * @param nx     Grid points per dimension [Nx, Ny, Nz]
     * @param lx     Box lengths [Lx, Ly, Lz]
     * @param bc     Boundary conditions (e.g., "periodic", "reflecting")
     * @param angles Lattice angles [alpha, beta, gamma] in degrees
     * @param mask   Optional mask for impenetrable regions
     * @return Pointer to created ComputationBox
     */
    virtual ComputationBox<T>* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        std::vector<double> angles,
        const double* mask=nullptr) = 0;

    /**
     * @brief Create a Molecules container.
     *
     * @param chain_model  "continuous" or "discrete"
     * @param ds           Contour step size
     * @param bond_lengths Map of monomer types to segment lengths squared
     * @return Molecules instance
     *
     * @example
     * @code
     * std::map<std::string, double> bonds = {{"A", 1.0}, {"B", 1.0}};
     * auto* mols = factory->create_molecules_information("continuous", 0.01, bonds);
     * mols->add_polymer(1.0, diblock_blocks);
     * @endcode
     */
    virtual Molecules* create_molecules_information(
        std::string chain_model, double ds, std::map<std::string, double> bond_lengths) = 0;

    /**
     * @brief Create a PropagatorComputationOptimizer.
     *
     * This is a non-virtual method as the optimizer is platform-independent.
     *
     * @param molecules                     Molecules container
     * @param aggregate_propagator_computation Enable propagator aggregation optimization
     * @return PropagatorComputationOptimizer instance
     *
     * @note Aggregation optimization is described in J. Chem. Theory Comput. 2025, 21, 3676.
     *       It can significantly speed up computation for mixtures of similar chains.
     */
    PropagatorComputationOptimizer* create_propagator_computation_optimizer(Molecules* molecules, bool aggregate_propagator_computation)
    {
        return new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
    };

    /**
     * @brief Create a pseudo-spectral propagator solver.
     *
     * Creates an FFT-based solver for the modified diffusion equation.
     * Uses RQM4 (4th-order Richardson extrapolation) for continuous chains.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Computation scheduler
     * @return Platform-specific PropagatorComputation solver
     *
     * @note Requires periodic boundary conditions.
     */
    virtual PropagatorComputation<T>* create_pseudospectral_solver(
        ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer) = 0;

    /**
     * @brief Create a real-space propagator solver.
     *
     * Creates a finite-difference solver using Crank-Nicolson method.
     * Supports non-periodic boundary conditions.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Computation scheduler
     * @return Platform-specific PropagatorComputation solver
     *
     * @note This is a beta feature. Supports reflecting and absorbing BCs.
     */
    virtual PropagatorComputation<T>* create_realspace_solver(
        ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer) = 0;

    /**
     * @brief Create an Anderson Mixing optimizer.
     *
     * @param n_var      Number of field variables
     * @param max_hist   Maximum history length
     * @param start_error Error threshold for switching to Anderson mixing
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     * @return Platform-specific AndersonMixing instance
     *
     * @example
     * @code
     * int n_var = 32 * 32 * 32;  // Total grid points
     * auto* am = factory->create_anderson_mixing(n_var, 20, 1e-1, 0.1, 0.1);
     * @endcode
     */
    virtual AndersonMixing<T>* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) = 0;

    /**
     * @brief Display platform and configuration information.
     *
     * Prints details about the platform (CPU/GPU), available resources,
     * and any relevant configuration settings.
     */
    virtual void display_info() = 0;
};
#endif
