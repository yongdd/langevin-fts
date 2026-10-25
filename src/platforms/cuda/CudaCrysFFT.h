/**
 * @file CudaCrysFFT.h
 * @brief CUDA Crystallographic FFT using DCT-II/III for Pmmm symmetry.
 *
 * DCT-II/III is implemented via CudaRealTransform (FCT algorithm).
 *
 * **Grid size:**
 *
 * Logical grid: Nx × Ny × Nz (full simulation box, must be even)
 * Physical grid: (Nx/2) × (Ny/2) × (Nz/2) (1/8 of logical grid)
 *
 * **Reference:**
 *
 * Qiang & Li, "Accelerated pseudo-spectral method of self-consistent field
 * theory via crystallographic fast Fourier transform", Macromolecules (2020)
 */

#ifndef CUDA_CRYS_FFT_H_
#define CUDA_CRYS_FFT_H_

#include <array>
#include <memory>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCommon.h"

// Forward declaration
class CudaRealTransform3D;

/// CrysFFT mode selection for CUDA solvers.
enum class CudaCrysFFTMode
{
    None,
    Recursive3m,
    PmmmDct,
    ObliqueZ
};

/**
 * @class CudaCrysFFTBase
 * @brief Abstract base for CUDA crystallographic FFT implementations.
 *
 * A solver keeps one object per stream. The k-space tables (Boltzmann factors,
 * stress multipliers) depend only on the grid, the cell and the coefficient, so
 * the objects made by clone_sharing_tables() share them: a table is built once,
 * by the first object that needs it, and read by all. Building and looking up
 * a table hold a mutex; set_cell_para() must not run while any stream computes.
 */
class CudaCrysFFTBase
{
public:
    virtual ~CudaCrysFFTBase() = default;
    virtual void set_cell_para(const std::array<double, 6>& cell_para) = 0;
    /// Select the Boltzmann table of this coefficient, building it when it is missing.
    virtual void set_contour_step(double ds) = 0;
    virtual void diffusion(double* d_q_in, double* d_q_out) = 0;
    virtual void diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream) = 0;
    virtual void set_stream(cudaStream_t stream) = 0;
    /// A transform of the same grid and cell for another stream: its own plans and
    /// work buffers, the tables of this object (it starts on the legacy default stream).
    virtual std::unique_ptr<CudaCrysFFTBase> clone_sharing_tables() const = 0;
};

/**
 * @class CudaCrysFFT
 * @brief CUDA crystallographic FFT using DCT-II/III.
 *
 * Physical grid: (Nx/2) × (Ny/2) × (Nz/2)
 */
class CudaCrysFFT : public CudaCrysFFTBase
{
private:
    std::array<int, 3> nx_logical_;    ///< Logical grid dimensions (Nx, Ny, Nz)
    std::array<int, 3> nx_physical_;   ///< Physical grid: (Nx/2, Ny/2, Nz/2)
    int M_logical_;                     ///< Total logical grid size
    int M_physical_;                    ///< Total physical grid size

    // Cell parameters (Lx, Ly, Lz, alpha, beta, gamma)
    std::array<double, 6> cell_para_;

    /// Device tables shared with the clones (see CudaCrysFFTBase); members other
    /// than the mutex are accessed with it held
    struct Tables
    {
        std::mutex mutex;
        std::array<double, 6> cell_para;                                         ///< Cell of the tables
        std::map<double, CudaDeviceBuffer<double>> boltzmann;                    ///< exp(-k² ds)/M_logical per ds
        std::array<CudaDeviceBuffer<double>, 8> axis_k2;                         ///< Σ k_d² per axis mask
        std::map<std::pair<int, double>, CudaDeviceBuffer<double>> axis_boltz_k2;   ///< exp(-k²·coeff)·Σ k_d² per (axis mask, coeff)
        std::vector<CudaDeviceBuffer<double>> spare;                             ///< Buffers of the tables dropped by the last cell change
    };
    std::shared_ptr<Tables> tables_;

    std::map<double, const double*> boltz_seen_;   ///< Tables of tables_->boltzmann this object has looked up (no lock per step)
    const double* d_boltz_current_;   ///< Table of ds_current_ (not owning)
    double ds_current_;

    // CudaRealTransform objects for DCT-II (forward) and DCT-III (backward)
    std::unique_ptr<CudaRealTransform3D> dct_forward_;
    std::unique_ptr<CudaRealTransform3D> dct_backward_;

    // Normalization factor
    double norm_factor_;
    cudaStream_t stream_{0};      ///< CUDA stream for execution

    /// Clone of `other` sharing its tables (see clone_sharing_tables()).
    CudaCrysFFT(const CudaCrysFFT& other, std::shared_ptr<Tables> tables);

    /// Boltzmann factors for given ds (tables_->mutex held).
    CudaDeviceBuffer<double> generateBoltzmann(double ds);

    /// Device table of M_physical_ values from h (a spare buffer when one is left),
    /// copied on stream_ and complete on return (tables_->mutex held)
    CudaDeviceBuffer<double> uploadTable(const std::vector<double>& h);

    /// Compute per-axis k² arrays on the physical grid (host).
    void computeAxisK2Host(std::vector<double>& kx2, std::vector<double>& ky2, std::vector<double>& kz2) const;

public:
    /**
     * @brief Construct CudaCrysFFT for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     * @param trans_part (Ignored for DCT-II/III, kept for API compatibility)
     */
    CudaCrysFFT(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    ~CudaCrysFFT();

    /// Update cell parameters.
    void set_cell_para(const std::array<double, 6>& cell_para);

    /// Set contour step size.
    void set_contour_step(double ds);

    /**
     * @brief Apply diffusion operator using DCT-II/III.
     *
     * @param d_q_in  Input field on device (physical grid)
     * @param d_q_out Output field on device (physical grid)
     */
    void diffusion(double* d_q_in, double* d_q_out);
    void diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream);
    void set_stream(cudaStream_t stream);
    std::unique_ptr<CudaCrysFFTBase> clone_sharing_tables() const;

    /**
     * @brief Apply an arbitrary k-space multiplier: DCT-II -> multiply -> DCT-III.
     *
     * Round-trip normalization (1/M_logical) is applied internally, so the
     * multiplier array must contain raw (unnormalized) values, mirroring the
     * CPU CrysFFTPmmmBase::apply_multiplier convention.
     * The input array is preserved when d_q_in != d_q_out.
     *
     * @param d_q_in       Input field on device (physical grid)
     * @param d_q_out      Output field on device (physical grid)
     * @param d_multiplier Device multiplier array (physical grid)
     * @param stream       CUDA stream for execution
     */
    void apply_multiplier(double* d_q_in, double* d_q_out, const double* d_multiplier, cudaStream_t stream);

    /**
     * @brief Device array of Cartesian Σ k_d² over some axes on the physical DCT grid.
     *
     * Used for the CrysFFT stress fast path (continuous chains).
     * Cached; invalidated when the cell parameters change.
     *
     * @param axes Bit mask of the axes, 1=x, 2=y, 4=z (for example 7: |k|²)
     */
    const double* get_axis_k2_multiplier(int axes);

    /**
     * @brief Device array of exp(-k²·coeff)·Σ k_d² over some axes on the physical DCT grid.
     *
     * Used for the CrysFFT stress fast path (discrete chains).
     * Cached per (axes, coeff); invalidated when the cell parameters change.
     *
     * @param axes  Bit mask of the axes, 1=x, 2=y, 4=z
     * @param coeff Bond factor coefficient (b²·ds/6)
     */
    const double* get_axis_boltz_k2_multiplier(int axes, double coeff);

    // Getters
    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
};

#endif
