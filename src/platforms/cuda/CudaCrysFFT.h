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
#include "CudaCrysFFTTables.h"

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
 * the objects made by clone_sharing_tables() share them and the cell
 * (CudaCrysFFTTables.h): a table is built once, by the first object that needs
 * it, and set_cell_para() on any object changes the cell of all of them. It
 * must not run while any stream computes.
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

    /// Tables of M_physical_ values: exp(-k² ds)/M_logical per ds, Σ k_d² per axis
    /// mask and exp(-k²·coeff)·Σ k_d² per (axis mask, coeff), shared with the
    /// clones together with the cell (see CudaCrysFFTTables.h)
    using Table = CudaDeviceBuffer<double>;
    enum TableKind { kBoltzmann, kAxisK2, kAxisBoltzK2 };
    CudaCrysFFTTableView<Table> tables_;

    const double* d_boltz_current_;   ///< Table of ds_current_ (not owning)
    double ds_current_;

    // CudaRealTransform objects for DCT-II (forward) and DCT-III (backward)
    std::unique_ptr<CudaRealTransform3D> dct_forward_;
    std::unique_ptr<CudaRealTransform3D> dct_backward_;

    // Normalization factor
    double norm_factor_;
    cudaStream_t stream_{0};      ///< CUDA stream for execution

    /// Clone of `other` sharing its tables (see clone_sharing_tables()).
    explicit CudaCrysFFT(const CudaCrysFFT& other);

    /// Boltzmann factors for given ds and cell, in `buffers` (a table build, see CudaCrysFFTTables::get()).
    Table generateBoltzmann(double ds, const std::array<double, 6>& cell, Table buffers);

    /// The M_physical_ values of h in `buffers` (allocated when empty), copied
    /// on stream_ and complete on return
    Table uploadTable(const std::vector<double>& h, Table buffers);

    /// Compute per-axis k² arrays of the cell on the physical grid (host).
    void computeAxisK2Host(const std::array<double, 6>& cell,
                           std::vector<double>& kx2, std::vector<double>& ky2, std::vector<double>& kz2) const;

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
