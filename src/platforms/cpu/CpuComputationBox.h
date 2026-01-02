/**
 * @file CpuComputationBox.h
 * @brief CPU implementation of ComputationBox.
 *
 * This header provides CpuComputationBox, the CPU-specific implementation of
 * the ComputationBox interface. It manages the simulation grid and provides
 * integration methods using standard CPU memory.
 *
 * For CPU, this is essentially a thin wrapper around the base ComputationBox
 * class since all base class methods already operate on host memory.
 *
 * @see ComputationBox for the interface definition
 * @see CudaComputationBox for GPU implementation
 */
#ifndef CPU_COMPUTATION_BOX_H_
#define CPU_COMPUTATION_BOX_H_

#include <array>
#include <vector>
#include <cassert>

#include "Exception.h"
#include "ComputationBox.h"

/**
 * @class CpuComputationBox
 * @brief CPU-specific computation box implementation.
 *
 * Inherits from ComputationBox and uses the base class implementations
 * directly since they already operate on host memory. This class exists
 * to maintain the factory pattern and enable polymorphic usage.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * @note All methods from ComputationBox work directly with CPU arrays.
 *       No data transfers are needed unlike the GPU version.
 */
template <typename T>
class CpuComputationBox : public ComputationBox<T>
{
public:
    /**
     * @brief Construct a CPU computation box.
     *
     * Delegates to base class constructor. All parameters are stored
     * in host memory.
     *
     * @param nx   Grid dimensions [Nx, Ny, Nz]
     * @param lx   Box lengths [Lx, Ly, Lz]
     * @param bc   Boundary conditions
     * @param mask Optional mask array for impenetrable regions
     */
    CpuComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr)
        : ComputationBox<T>(nx, lx, bc, mask) {};

    /**
     * @brief Virtual destructor.
     */
    virtual ~CpuComputationBox() {};
};
#endif
