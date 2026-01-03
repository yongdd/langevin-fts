/**
 * @file CpuSolverPseudoBase.h
 * @brief Common base class for CPU pseudo-spectral solvers.
 *
 * This header provides CpuSolverPseudoBase, a common base class that
 * consolidates shared functionality between CpuSolverPseudoContinuous
 * and CpuSolverPseudoDiscrete, including:
 *
 * - FFT object management (creation/destruction)
 * - Pseudo-spectral operator helper
 * - Boundary condition handling
 * - Transform dispatch methods
 * - Laplacian operator updates
 * - Stress computation (with customizable coefficient function)
 *
 * **Design Decision:**
 *
 * All simulations are conducted in fixed dimensions (1D, 2D, or 3D).
 * The FFT object is stored as `FFT<T>* fft_` which points to MklFFT<T, DIM>
 * based on the dimensionality. This avoids maintaining separate fft_1d,
 * fft_2d, fft_3d pointers when only one is ever used per solver.
 *
 * @see CpuSolverPseudoContinuous for continuous chain implementation
 * @see CpuSolverPseudoDiscrete for discrete chain implementation
 */

#ifndef CPU_SOLVER_PSEUDO_BASE_H_
#define CPU_SOLVER_PSEUDO_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <complex>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "MklFFT.h"

/**
 * @class CpuSolverPseudoBase
 * @brief Common base class for CPU pseudo-spectral propagator solvers.
 *
 * Consolidates shared code between continuous and discrete chain solvers,
 * including FFT management, transform dispatch, and stress computation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Shared Functionality:**
 *
 * - FFT object creation/destruction with dimension dispatch
 * - Forward/backward transform methods
 * - Laplacian operator updates
 * - Stress computation framework
 *
 * **Virtual Methods:**
 *
 * Derived classes must implement:
 * - update_dw(): Different Boltzmann factor formulas
 * - advance_propagator(): Different propagation algorithms
 * - advance_propagator_half_bond_step(): Discrete-specific
 * - get_stress_boltz_bond(): Coefficient factor for stress
 */
template <typename T>
class CpuSolverPseudoBase : public CpuSolver<T>
{
protected:
    ComputationBox<T>* cb;       ///< Computation box for grid info
    Molecules* molecules;        ///< Molecules container
    std::string chain_model;     ///< Chain model identifier

    /**
     * @brief Spectral transform object for FFT/DCT/DST (handles all BCs).
     *
     * Points to MklFFT<T, 1>, MklFFT<T, 2>, or MklFFT<T, 3> based on dim_.
     * All simulations are conducted in fixed dimensions, so only one
     * transform object is needed per solver instance.
     *
     * Uses FFT<T>* base class pointer to enable polymorphic calls without
     * dimension-specific casting or if-else dispatch.
     */
    FFT<T>* fft_;

    Pseudo<T>* pseudo;           ///< Pseudo-spectral operator helper

    bool is_periodic_;           ///< True if all BCs are periodic
    int dim_;                    ///< Number of dimensions (1, 2, or 3)

    /**
     * @brief Perform forward transform.
     *
     * Delegates to fft_->forward() which handles all boundary conditions.
     *
     * @param rdata Real-space data input
     * @param cdata Fourier/DCT coefficient output
     */
    void transform_forward(T* rdata, double* cdata) { fft_->forward(rdata, cdata); }

    /**
     * @brief Perform backward transform.
     *
     * Delegates to fft_->backward() which handles all boundary conditions.
     *
     * @param cdata Fourier/DCT coefficient input
     * @param rdata Real-space data output
     */
    void transform_backward(double* cdata, T* rdata) { fft_->backward(cdata, rdata); }

    /**
     * @brief Initialize shared components in constructor.
     *
     * Sets up:
     * - Boundary condition checking (is_periodic_)
     * - FFT object creation (fft_)
     * - Pseudo-spectral operator (pseudo)
     *
     * @param cb Computation box
     * @param molecules Molecules container
     */
    void init_shared(ComputationBox<T>* cb, Molecules* molecules);

    /**
     * @brief Clean up shared components in destructor.
     *
     * Properly deletes FFT object with correct type cast and Pseudo object.
     */
    void cleanup_shared();

    /**
     * @brief Get Boltzmann bond factor for stress computation.
     *
     * Returns the appropriate multiplier for stress coefficient:
     * - Continuous: Returns nullptr (no boltz_bond factor in stress)
     * - Discrete: Returns boltz_bond or boltz_bond_half based on is_half_bond_length
     *
     * @param monomer_type Monomer type
     * @param is_half_bond_length Whether using half bond length
     * @return Pointer to Boltzmann bond array, or nullptr if not applicable
     */
    virtual const double* get_stress_boltz_bond(
        std::string monomer_type, bool is_half_bond_length) const = 0;

public:
    /**
     * @brief Default constructor.
     */
    CpuSolverPseudoBase() : fft_(nullptr), pseudo(nullptr), is_periodic_(true), dim_(0) {}

    /**
     * @brief Virtual destructor.
     */
    virtual ~CpuSolverPseudoBase() {}

    /**
     * @brief Update Fourier-space diffusion operators.
     *
     * Recomputes exp(-k^2 b^2 ds/6) for each monomer type when box
     * dimensions change. Required after box size updates.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Compute stress contribution from a single segment.
     *
     * Common implementation that calls get_stress_boltz_bond() for
     * chain-model-specific coefficient factor.
     *
     * @param q_1                Forward propagator
     * @param q_2                Backward propagator
     * @param monomer_type       Monomer type for segment length
     * @param is_half_bond_length Whether using half bond (discrete model)
     *
     * @return Stress components [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz]
     */
    std::vector<T> compute_single_segment_stress(
        T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length) override;
};

#endif
