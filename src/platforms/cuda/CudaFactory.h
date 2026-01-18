/**
 * @file CudaFactory.h
 * @brief Concrete factory for CUDA GPU-based implementations.
 *
 * This header provides CudaFactory, the concrete implementation of
 * AbstractFactory for GPU-accelerated computations using NVIDIA CUDA.
 * It creates CUDA-optimized versions of all simulation objects.
 *
 * **Platform Features:**
 *
 * - Uses cuFFT for GPU-accelerated FFT operations
 * - Multiple CUDA streams for parallel propagator computation
 * - Asynchronous memory transfers to hide latency
 * - Supports both continuous and discrete chain models
 * - Checkpointing mode stores propagators in pinned host memory
 *
 * **Performance Notes:**
 *
 * - Best for 2D and 3D simulations with large grids (> 32³)
 * - GPU memory can be a limiting factor for large systems
 * - Use use_checkpointing=true for large grids with limited GPU RAM
 *
 * **Memory Modes:**
 *
 * - Standard mode: All propagators stored in GPU memory (fastest)
 * - Checkpointing mode: Propagators stored in pinned host memory with
 *   async transfers (2-4× slower but uses much less GPU memory)
 *
 * @see AbstractFactory for the factory interface
 * @see MklFactory for CPU implementation
 * @see PlatformSelector for factory instantiation
 *
 * @example
 * @code
 * // Create CUDA factory (use_checkpointing=false for speed)
 * CudaFactory<double> factory(false);
 *
 * // Create simulation objects
 * auto* cb = factory.create_computation_box(nx, lx, bc);
 * auto* solver = factory.create_propagator_computation(cb, mols, optimizer, "rqm4");
 *
 * // Display GPU info
 * factory.display_info();
 * @endcode
 */

#ifndef CUDA_FACTORY_H_
#define CUDA_FACTORY_H_

#include "ComputationBox.h"
#include "Polymer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "Array.h"

/**
 * @class CudaFactory
 * @brief Factory for creating CUDA GPU-accelerated simulation objects.
 *
 * Implements AbstractFactory to create GPU-optimized objects using CUDA.
 * All created objects operate on GPU device memory with cuFFT for FFTs.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Created Object Types:**
 *
 * - CudaComputationBox: Grid and integration on GPU
 * - CudaComputationContinuous/Discrete: GPU propagator solvers
 * - CudaAndersonMixing: GPU iteration accelerator
 *
 * **Memory Management:**
 *
 * When use_checkpointing=true:
 * - Uses CudaComputationReduceMemoryContinuous/Discrete
 * - Uses CudaAndersonMixingReduceMemory
 * - Stores propagator history in pinned host memory
 * - Overlaps kernel execution with async memory transfers
 */
template <typename T>
class CudaFactory : public AbstractFactory<T>
{
private:
    bool checkpoint_on_host;  ///< If true, store checkpoints in pinned host memory; if false, use GPU global memory

public :
    /**
     * @brief Construct a CUDA factory.
     *
     * @param use_checkpointing If true, use checkpointing mode that stores only
     *                          propagator checkpoints instead of full histories,
     *                          recomputing as needed (reduces memory, increases compute)
     * @param checkpoint_on_host If true (default) and use_checkpointing is true,
     *                           store checkpoints in pinned host memory.
     *                           If false, store in GPU global memory for faster
     *                           access but higher GPU memory usage.
     */
    CudaFactory(bool use_checkpointing, bool checkpoint_on_host = true);

    // Array* create_array(
    //     unsigned int size) override;

    // Array* create_array(
    //     double *data,
    //     unsigned int size) override;

    /**
     * @brief Create a CUDA computation box.
     *
     * @param nx   Grid dimensions [Nx, Ny, Nz]
     * @param lx   Box lengths [Lx, Ly, Lz]
     * @param bc   Boundary conditions ("periodic", "reflecting", "absorbing")
     * @param mask Optional mask for impenetrable regions
     * @return CudaComputationBox instance
     */
    ComputationBox<T>* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        const double* mask=nullptr) override;

    /**
     * @brief Create a CUDA computation box with lattice angles.
     *
     * @param nx     Grid dimensions [Nx, Ny, Nz]
     * @param lx     Box lengths [Lx, Ly, Lz]
     * @param bc     Boundary conditions
     * @param angles Lattice angles [alpha, beta, gamma] in degrees
     * @param mask   Optional mask for impenetrable regions
     * @return CudaComputationBox instance
     */
    ComputationBox<T>* create_computation_box(
        std::vector<int> nx,
        std::vector<double> lx,
        std::vector<std::string> bc,
        std::vector<double> angles,
        const double* mask=nullptr) override;

    /**
     * @brief Create molecules container.
     *
     * @param chain_model  "continuous" or "discrete"
     * @param ds           Contour step size
     * @param bond_lengths Segment lengths squared by monomer type
     * @return Molecules instance
     */
    Molecules* create_molecules_information(
        std::string chain_model, double ds, std::map<std::string, double> bond_lengths) override;

    /**
     * @brief Create CUDA propagator computation solver.
     *
     * Creates CudaComputationContinuous, CudaComputationDiscrete, or their
     * reduce-memory variants based on chain model, memory mode, and numerical method.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Computation scheduler
     * @param numerical_method                "rqm4", "etdrk4" (pseudo-spectral) or
     *                                        "cn-adi2", "cn-adi4-lr" (real-space)
     * @return CUDA propagator solver
     */
    PropagatorComputation<T>* create_propagator_computation(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string numerical_method) override;

    /**
     * @brief Create CUDA Anderson mixing optimizer.
     *
     * Creates CudaAndersonMixing or CudaAndersonMixingReduceMemory
     * based on the use_checkpointing setting.
     *
     * @param n_var      Number of field variables
     * @param max_hist   Maximum history length
     * @param start_error Error threshold for Anderson mixing
     * @param mix_min    Minimum mixing parameter
     * @param mix_init   Initial mixing parameter
     * @return CUDA Anderson mixing instance
     */
    AndersonMixing<T>* create_anderson_mixing(
        int n_var, int max_hist, double start_error,
        double mix_min, double mix_init) override;

    /**
     * @brief Display CUDA platform information.
     *
     * Prints GPU device name, compute capability, memory, and
     * other configuration details.
     */
    void display_info() override;
};
#endif
