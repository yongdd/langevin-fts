/**
 * @file MklCrysFFTPmmm.h
 * @brief MKL Crystallographic FFT using DCT-II/III for Pmmm symmetry.
 *
 * For Pmmm symmetry (3 perpendicular mirrors), DCT-II/III provides exact
 * pseudo-spectral diffusion with 1/8 the computational cost.
 *
 * **Algorithm:**
 *
 * DCT-II/III solves the diffusion equation with Neumann boundary conditions
 * at half-grid points:
 *   q_out(r) = DCT-III[ exp(-k²ds) * DCT-II[q_in(r)] ]
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
 *
 * @see MklFFT for standard FFT implementation
 * @see FftwCrysFFTPmmm for FFTW version
 */

#ifndef MKL_CRYS_FFT_PMMM_H_
#define MKL_CRYS_FFT_PMMM_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include <complex>
#include "mkl_dfti.h"

/**
 * @class MklCrysFFTPmmm
 * @brief MKL crystallographic FFT using DCT-II/III via FFT.
 *
 * Uses DCT-II (forward) and DCT-III (backward) for Pmmm symmetry,
 * implemented via O(N log N) FFT algorithm.
 *
 * **Memory Layout:**
 *
 * - Logical grid: Nx × Ny × Nz (must be even in all dimensions)
 * - Physical grid: (Nx/2) × (Ny/2) × (Nz/2)
 * - Boltzmann factors: one real array per ds value
 */
class MklCrysFFTPmmm
{
private:
    std::array<int, 3> nx_logical_;    ///< Logical grid dimensions (Nx, Ny, Nz)
    std::array<int, 3> nx_physical_;   ///< Physical grid: (Nx/2, Ny/2, Nz/2)
    int M_logical_;                     ///< Total logical grid size
    int M_physical_;                    ///< Total physical grid size

    // Cell parameters (Lx, Ly, Lz, alpha, beta, gamma)
    std::array<double, 6> cell_para_;

    struct BoltzDeleter {
        void operator()(double* ptr) const { delete[] ptr; }
    };
    struct ThreadState
    {
        std::map<double, std::unique_ptr<double, BoltzDeleter>> boltzmann;
        const double* boltz_current = nullptr;
        double ds_current = std::numeric_limits<double>::quiet_NaN();
        uint64_t epoch = 0;
        uint64_t instance_id = 0;
    };
    inline static std::atomic<uint64_t> next_instance_id_{1};
    uint64_t instance_id_{0};
    mutable std::atomic<uint64_t> cache_epoch_{1};

    // MKL DFTI descriptors for DCT via FFT (per dimension)
    std::array<DFTI_DESCRIPTOR_HANDLE, 3> dct_fft_forward_;   ///< Forward FFT for DCT-II
    std::array<DFTI_DESCRIPTOR_HANDLE, 3> dct_fft_backward_;  ///< Backward FFT for DCT-III

    // Precomputed twiddle factors for O(N log N) DCT via FFT
    std::array<std::vector<double>, 3> cos_tables_;
    std::array<std::vector<double>, 3> sin_tables_;

    // Work buffers (allocated once for plan creation)
    double* io_buffer_;    ///< I/O buffer for physical grid
    double* temp_buffer_;  ///< Temp buffer for DCT output

    // Normalization factor for DCT-II/III round trip: 1/M_logical
    double norm_factor_;

    /**
     * @brief Generate Boltzmann factors for given ds.
     * @param ds Contour step size
     */
    double* generateBoltzmann(double ds) const;

    /**
     * @brief Initialize MKL DFTI descriptors for DCT via FFT.
     */
    void initFFTPlans();

    /**
     * @brief Precompute twiddle factors for DCT via FFT.
     */
    void precomputeTwiddleFactors();

    /**
     * @brief Free Boltzmann factor memory.
     */
    void freeBoltzmann();
    ThreadState& get_thread_state() const;

    /**
     * @brief Apply DCT-II forward along all dimensions.
     */
    void applyDCT2Forward3D(double* data, double* temp);

    /**
     * @brief Apply DCT-III backward along all dimensions.
     */
    void applyDCT3Backward3D(double* data, double* temp);

public:
    /**
     * @brief Construct MklCrysFFTPmmm for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     *                   Angles in radians. For cubic: all 90° = π/2.
     * @param trans_part (Ignored for DCT-II/III, kept for API compatibility)
     */
    MklCrysFFTPmmm(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    /**
     * @brief Destructor. Frees memory and MKL descriptors.
     */
    ~MklCrysFFTPmmm();

    /**
     * @brief Update cell parameters.
     *
     * Invalidates all Boltzmann factors (they will be regenerated on next use).
     *
     * @param cell_para New cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     */
    void set_cell_para(const std::array<double, 6>& cell_para);

    /**
     * @brief Set contour step size and prepare Boltzmann factors.
     *
     * If Boltzmann factors for this ds don't exist, they will be generated.
     * Otherwise, cached factors will be used.
     *
     * @param ds Contour step size
     */
    void set_contour_step(double ds);

    /**
     * @brief Apply diffusion operator using DCT-II/III.
     *
     * Computes: q_out = DCT-III[ exp(-k²ds) * DCT-II[q_in] ]
     *
     * Input and output are on the PHYSICAL grid ((N/2)³).
     *
     * @param q_in  Input field on physical grid
     * @param q_out Output field on physical grid
     */
    void diffusion(double* q_in, double* q_out);

    /**
     * @brief Apply custom multiplier in Fourier space using DCT-II/III.
     *
     * Computes: q_out = DCT-III[ multiplier * DCT-II[q_in] ]
     *
     * Input/output and multiplier are on the PHYSICAL grid ((N/2)³).
     *
     * @param q_in       Input field on physical grid
     * @param q_out      Output field on physical grid
     * @param multiplier Real multiplier on physical grid (same size as q_in)
     */
    void apply_multiplier(const double* q_in, double* q_out, const double* multiplier);

    /**
     * @brief Get logical grid dimensions.
     */
    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }

    /**
     * @brief Get physical grid dimensions ((N/2)³).
     */
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }

    /**
     * @brief Get logical grid size.
     */
    int get_M_logical() const { return M_logical_; }

    /**
     * @brief Get physical grid size.
     */
    int get_M_physical() const { return M_physical_; }

    /**
     * @brief Get I/O buffer pointer.
     */
    double* get_io_buffer() { return io_buffer_; }
};

#endif  // MKL_CRYS_FFT_PMMM_H_
