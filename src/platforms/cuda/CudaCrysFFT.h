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
#include <map>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

// Forward declaration
class CudaRealTransform3D;

/**
 * @brief CrysFFT mode selection for CUDA solvers.
 */
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
 */
class CudaCrysFFTBase
{
public:
    virtual ~CudaCrysFFTBase() = default;
    virtual void set_cell_para(const std::array<double, 6>& cell_para) = 0;
    virtual void set_contour_step(double ds) = 0;
    virtual void diffusion(double* d_q_in, double* d_q_out) = 0;
    virtual void diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream) = 0;
    virtual void set_stream(cudaStream_t stream) = 0;
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

    // Boltzmann factors - per ds value (device memory)
    std::map<double, double*> d_boltzmann_;
    double* d_boltz_current_;
    double ds_current_;

    // CudaRealTransform objects for DCT-II (forward) and DCT-III (backward)
    CudaRealTransform3D* dct_forward_;
    CudaRealTransform3D* dct_backward_;

    // Device buffers
    double* d_work_;              ///< Work buffer (physical grid)

    // Normalization factor
    double norm_factor_;
    cudaStream_t stream_{0};      ///< CUDA stream for execution

    // Cached stress multipliers (device memory)
    double* d_axis_k2_[3] = {nullptr, nullptr, nullptr};          ///< k_axis² per axis
    std::map<std::pair<int, double>, double*> d_axis_boltz_k2_;   ///< exp(-k²·coeff)·k_axis² per (axis, coeff)

    /**
     * @brief Generate Boltzmann factors for given ds.
     */
    void generateBoltzmann(double ds);

    /**
     * @brief Free Boltzmann factor memory.
     */
    void freeBoltzmann();

    /**
     * @brief Free cached stress multiplier arrays.
     */
    void freeMultiplierCaches();

    /**
     * @brief Compute per-axis k² arrays on the physical grid (host).
     */
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

    /**
     * @brief Update cell parameters.
     */
    void set_cell_para(const std::array<double, 6>& cell_para);

    /**
     * @brief Set contour step size.
     */
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
     * @brief Device array of Cartesian k_axis² on the physical DCT grid.
     *
     * Used for the CrysFFT stress fast path (continuous chains).
     * Cached; invalidated when the cell parameters change.
     *
     * @param axis 0=x, 1=y, 2=z
     */
    const double* get_axis_k2_multiplier(int axis);

    /**
     * @brief Device array of exp(-k²·coeff)·k_axis² on the physical DCT grid.
     *
     * Used for the CrysFFT stress fast path (discrete chains).
     * Cached per (axis, coeff); invalidated when the cell parameters change.
     *
     * @param axis  0=x, 1=y, 2=z
     * @param coeff Bond factor coefficient (b²·ds/6)
     */
    const double* get_axis_boltz_k2_multiplier(int axis, double coeff);

    // Getters
    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
};

#endif
