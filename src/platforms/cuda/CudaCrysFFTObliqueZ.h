/**
 * @file CudaCrysFFTObliqueZ.h
 * @brief CUDA CrysFFT for z-mirror symmetry (DCT-z + FFT-xy, gamma arbitrary).
 *
 * The DCT-II along z and the FFT along x, y run as one real 3D FFT of the
 * physical grid (Nx, Ny, Nz/2) with Makhoul's z permutation (even samples
 * forward, odd samples backward). One kernel then forms the DCT coefficients
 * of the pair kz, Nz/2 - kz from the half spectrum, applies their Boltzmann
 * factors and returns to the half spectrum of the inverse permutation:
 *
 *   P'(k) = S(k) P(k) + D(k) e^{i pi kz/(Nz/2)} conj(P(-kx, -ky, kz)),
 *
 * with S = B(kz) + B(Nz/2 - kz), D = B(kz) - B(Nz/2 - kz) and B(Nz/2) = 0.
 * This needs the Boltzmann factors to be even in (kx, ky), which holds for
 * alpha = beta = 90 degrees.
 */

#ifndef CUDA_CRYS_FFT_OBLIQUEZ_H_
#define CUDA_CRYS_FFT_OBLIQUEZ_H_

#include <array>
#include <memory>
#include <map>
#include <mutex>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCrysFFT.h"

class CudaCrysFFTObliqueZ : public CudaCrysFFTBase
{
private:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_spectral_{0};   ///< Nx * Ny * (Nz/4 + 1): half spectrum of the physical grid

    /// (S, D) pairs per ds (see the file comment), shared with the clones
    /// together with the cell (see CudaCrysFFTTables.h)
    using Table = CudaDeviceBuffer<double>;
    enum TableKind { kBoltzmann };
    CudaCrysFFTTableView<Table> tables_;

    const double* d_boltz_current_{nullptr};   ///< Table of ds_current_ (not owning)
    double ds_current_{0.0};

    cufftHandle plan_fwd_{};   ///< 3D D2Z of the physical grid
    cufftHandle plan_bwd_{};   ///< 3D Z2D of the physical grid
    bool plans_initialized_{false};

    /// (cos, sin)(pi kz/(Nz/2)) per stored kz; depends only on the grid, shared with the clones
    std::shared_ptr<const CudaDeviceBuffer<double>> d_twiddle_;
    CudaDeviceBuffer<double> d_real_;                  ///< Permuted physical grid
    CudaDeviceBuffer<cufftDoubleComplex> d_spectrum_;  ///< Its half spectrum (updated in place)

    double norm_factor_{1.0};
    cudaStream_t stream_{0};

    /// Clone of `other` sharing its twiddle factors and tables (see clone_sharing_tables()).
    explicit CudaCrysFFTObliqueZ(const CudaCrysFFTObliqueZ& other);

    /// Create the 3D plans, the work buffers and (unless shared) the twiddle factors.
    void init_transforms();
    void free_plans();
    /// (S, D) table for given ds and cell, in `buffers` (a table build, see CudaCrysFFTTables::get()).
    Table generateBoltzmann(double ds, const std::array<double, 6>& cell, Table buffers);
    /// Free everything acquired so far (idempotent; also used when the constructor throws).
    void release();

public:
    CudaCrysFFTObliqueZ(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    ~CudaCrysFFTObliqueZ() override;

    void set_cell_para(const std::array<double, 6>& cell_para) override;
    void set_contour_step(double ds) override;
    void diffusion(double* d_q_in, double* d_q_out) override;
    void diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream) override;
    void set_stream(cudaStream_t stream) override;
    std::unique_ptr<CudaCrysFFTBase> clone_sharing_tables() const override;

    int get_M_physical() const { return M_physical_; }
};

#endif  // CUDA_CRYS_FFT_OBLIQUEZ_H_
