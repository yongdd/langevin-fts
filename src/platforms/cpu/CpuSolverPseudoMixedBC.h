/**
 * @file CpuSolverPseudoMixedBC.h
 * @brief Pseudo-spectral solver with mixed boundary conditions on CPU.
 *
 * This header provides CpuSolverPseudoMixedBC, which implements the
 * pseudo-spectral method with support for mixed boundary conditions:
 * - PERIODIC: Standard FFT
 * - REFLECTING: DCT-II/III (Neumann BC, zero-flux at boundaries)
 * - ABSORBING: DST-II/III (Dirichlet BC, zero value at boundaries)
 *
 * **Physical Meaning:**
 *
 * - REFLECTING: Models impenetrable walls where polymer chains cannot cross
 * - ABSORBING: Models surfaces that absorb polymer chain ends
 *
 * @see CpuSolverPseudoContinuous for periodic-only solver
 * @see MklFFTMixedBC for the FFT implementation
 * @see PseudoMixedBC for Boltzmann factor computation
 */

#ifndef CPU_SOLVER_PSEUDO_MIXED_BC_H_
#define CPU_SOLVER_PSEUDO_MIXED_BC_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "MklFFT.h"
#include "MklFFTMixedBC.h"

/**
 * @class CpuSolverPseudoMixedBC
 * @brief CPU pseudo-spectral solver with mixed boundary conditions.
 *
 * Supports reflecting (DCT) and absorbing (DST) boundaries in addition
 * to periodic (FFT). Currently, mixed BC per dimension is not yet supported;
 * use all periodic or all non-periodic.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 */
template <typename T>
class CpuSolverPseudoMixedBC : public CpuSolver<T>
{
private:
    ComputationBox<T>* cb;           ///< Computation box for grid info
    Molecules* molecules;             ///< Molecules container
    std::string chain_model;          ///< Chain model identifier

    // FFT objects - one of these will be used based on BC
    FFT<T>* fft_periodic;             ///< Standard FFT for periodic BC
    MklFFTMixedBC<T, 1>* fft_mixed_1d;  ///< Mixed BC FFT for 1D
    MklFFTMixedBC<T, 2>* fft_mixed_2d;  ///< Mixed BC FFT for 2D
    MklFFTMixedBC<T, 3>* fft_mixed_3d;  ///< Mixed BC FFT for 3D

    Pseudo<T>* pseudo;                ///< Pseudo-spectral operator helper

    bool is_periodic_;                ///< True if all BCs are periodic

    /**
     * @brief Perform forward transform based on BC type.
     */
    void transform_forward(T* rdata, double* cdata);

    /**
     * @brief Perform backward transform based on BC type.
     */
    void transform_backward(double* cdata, T* rdata);

public:
    /**
     * @brief Construct pseudo-spectral solver with mixed BCs.
     *
     * @param cb        Computation box defining the grid and BCs
     * @param molecules Molecules container with monomer types
     */
    CpuSolverPseudoMixedBC(ComputationBox<T>* cb, Molecules* molecules);

    /**
     * @brief Destructor.
     */
    ~CpuSolverPseudoMixedBC();

    /**
     * @brief Update Fourier-space diffusion operators.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors from potential fields.
     */
    void update_dw(std::map<std::string, const T*> w_input) override;

    /**
     * @brief Advance propagator by one contour step ds.
     *
     * Uses 4th-order Richardson extrapolation.
     */
    void advance_propagator(T* q_in, T* q_out, std::string monomer_type, const double* q_mask) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     */
    void advance_propagator_half_bond_step(T*, T*, std::string) override {};

    /**
     * @brief Compute stress contribution from one segment.
     */
    std::vector<T> compute_single_segment_stress(
        T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length) override;
};

#endif
