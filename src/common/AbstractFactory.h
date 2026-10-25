/**
 * @file AbstractFactory.h
 * @brief Abstract Factory pattern for platform-independent object creation.
 *
 * This header defines the AbstractFactory class, which implements the Abstract
 * Factory design pattern to create platform-specific objects (CPU/FFTW or CUDA)
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
 * - **cpu-mkl**: Intel MKL-based CPU implementation (default)
 * - **cpu-fftw**: FFTW-based CPU implementation using OpenMP
 * - **cuda**: NVIDIA CUDA GPU implementation using cuFFT
 *
 * @see MklFactory for CPU implementation (MKL)
 * @see FftwFactory for CPU implementation (FFTW)
 * @see CudaFactory for GPU implementation
 * @see PlatformSelector for factory creation
 *
 * @example
 * @code
 * // Create factory for the desired platform
 * std::unique_ptr<AbstractFactory<double>> factory = PlatformSelector::create_factory_real(
 *     "cuda", false);  // Use CUDA, no memory reduction
 *
 * // Create simulation objects through factory
 * std::unique_ptr<ComputationBox<double>> cb = factory->create_computation_box(nx, lx, bc);
 * std::unique_ptr<Molecules> molecules = factory->create_molecules_information("continuous", ds, bonds);
 * auto optimizer = factory->create_propagator_computation_optimizer(molecules.get(), true);
 * auto solver = factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), "rqm4");
 * auto am = factory->create_anderson_mixing(n_var, 20, 1e-1, 0.1, 0.1);
 *
 * // Display platform information
 * factory->display_info();
 *
 * // The objects are destroyed in reverse order of creation, so the solver
 * // goes before the box, molecules and optimizer it points to.
 * @endcode
 *
 * Every create function returns a std::unique_ptr. An object created from
 * others keeps raw pointers to them, so they must outlive it.
 */

#ifndef ABSTRACT_FACTORY_H_
#define ABSTRACT_FACTORY_H_

#include <string>
#include <array>
#include <complex>
#include <memory>

#include "ComputationBox.h"
#include "ComputationBackend.h"
#include "NumericalMethods.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "SpaceGroup.h"

/**
 * @class AbstractFactory
 * @brief Abstract factory for creating platform-specific simulation objects.
 *
 * This template class defines the interface for creating all major objects
 * needed for polymer field theory simulations. Concrete implementations
 * (MklFactory, FftwFactory, CudaFactory) provide platform-optimized versions of
 * each object. The propagator computation is assembled here, the same way on
 * every platform; a platform contributes only its backend (create_backend()).
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
 * The reduce_memory flag enables checkpointing mode where only
 * propagator checkpoints are stored. This increases computation time
 * 2-4x but significantly reduces memory usage.
 *
 * @see PlatformSelector::create_factory_real for factory instantiation
 */
template <typename T>
class AbstractFactory
{
protected:
    bool reduce_memory;    ///< Enable checkpointing mode (reduces memory, increases compute)

    /**
     * @brief Create the computation backend of the platform.
     *
     * The only platform-specific step of create_propagator_computation(). The
     * computation calls it once while it is constructed, after the space group
     * is set on the box.
     *
     * @param cb            Computation box
     * @param molecules     Molecules container
     * @param method_info   Numerical method, already resolved and validated
     * @param space_group   Space group (nullptr if none)
     * @param max_n_streams Upper limit of concurrent propagator streams
     */
    virtual std::unique_ptr<ComputationBackend<T>> create_backend(
        ComputationBox<T>* cb, Molecules* molecules, const NumericalMethodInfo& method_info,
        SpaceGroup* space_group, int max_n_streams) = 0;

public :
    /// Virtual destructor.
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
     * auto cb = factory->create_computation_box(nx, lx, bc);
     * @endcode
     */
    virtual std::unique_ptr<ComputationBox<T>> create_computation_box(
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
     * @param angles Lattice angles in degrees. Accepts dimension-appropriate input:
     *               - Empty: orthogonal system (all angles 90°)
     *               - 1 element: [gamma] for 2D (alpha=beta=90°)
     *               - 3 elements: [alpha, beta, gamma] for full specification
     * @param mask   Optional mask for impenetrable regions
     * @return Platform-specific ComputationBox instance
     */
    virtual std::unique_ptr<ComputationBox<T>> create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        std::vector<double> angles,
        const double* mask=nullptr) = 0;

    /**
     * @brief Create a Molecules container.
     *
     * This is a non-virtual method as the container is platform-independent.
     *
     * @param chain_model  "continuous" or "discrete"
     * @param ds           Contour step size
     * @param bond_lengths Map of monomer types to segment lengths squared
     * @return Molecules instance
     *
     * @example
     * @code
     * std::map<std::string, double> bonds = {{"A", 1.0}, {"B", 1.0}};
     * auto mols = factory->create_molecules_information("continuous", 0.01, bonds);
     * mols->add_polymer(1.0, diblock_blocks);
     * @endcode
     */
    std::unique_ptr<Molecules> create_molecules_information(
        std::string chain_model, double ds, std::map<std::string, double> bond_lengths)
    {
        return std::make_unique<Molecules>(chain_model, ds, bond_lengths);
    }

    /**
     * @brief Create a PropagatorComputationOptimizer.
     *
     * This is a non-virtual method as the optimizer is platform-independent.
     *
     * @param molecules                     Molecules container; must outlive the optimizer
     * @param aggregate_propagator_computation Enable propagator aggregation optimization
     * @return PropagatorComputationOptimizer instance
     *
     * @note Aggregation optimization is described in J. Chem. Theory Comput. 2025, 21, 3676.
     *       It can significantly speed up computation for mixtures of similar chains.
     */
    std::unique_ptr<PropagatorComputationOptimizer> create_propagator_computation_optimizer(Molecules* molecules, bool aggregate_propagator_computation)
    {
        return std::make_unique<PropagatorComputationOptimizer>(molecules, aggregate_propagator_computation);
    };

    /**
     * @brief Create a propagator computation solver.
     *
     * Creates a solver for the chain propagator computation. The numerical method
     * determines whether a pseudo-spectral (FFT-based) or real-space (finite-difference)
     * solver is used. The chain model and reduce_memory select the computation class
     * (ComputationContinuous, ComputationDiscrete or their ComputationReduceMemory*
     * variants); create_backend() supplies the platform. Implemented in
     * AbstractFactory.cpp for every platform.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Computation scheduler
     * @param numerical_method                Numerical method:
     *                                        - "rqm4": 4th-order Richardson (pseudo-spectral)
     *                                        - "rk2":  2nd-order operator splitting (pseudo-spectral)
     *                                        - "cn-adi2": 2nd-order Crank-Nicolson ADI (real-space)
     * @param space_group                     Optional space group (reduced basis, real fields only)
     * @return PropagatorComputation solver with the platform's backend. The box,
     *         molecules, optimizer and space group must outlive it.
     *
     * @note Real-space methods (cn-adi*) support non-periodic boundary conditions
     *       but are a beta feature.
     */
    std::unique_ptr<PropagatorComputation<T>> create_propagator_computation(
        ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer,
        std::string numerical_method, SpaceGroup* space_group=nullptr);

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
     * auto am = factory->create_anderson_mixing(n_var, 20, 1e-1, 0.1, 0.1);
     * @endcode
     */
    virtual std::unique_ptr<AndersonMixing<T>> create_anderson_mixing(
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
