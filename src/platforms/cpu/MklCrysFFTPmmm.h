/**
 * @file MklCrysFFTPmmm.h
 * @brief MKL implementation of crystallographic FFT for Pmmm symmetry.
 *
 * Uses O(N log N) FFT algorithm with twiddle factors to implement
 * DCT-II/III for Pmmm mirror symmetry.
 *
 * @see CrysFFTPmmmBase for common functionality
 * @see FftwCrysFFTPmmm for FFTW implementation
 */

#ifndef MKL_CRYS_FFT_PMMM_H_
#define MKL_CRYS_FFT_PMMM_H_

#include <array>
#include <tuple>
#include <vector>
#include <complex>
#include "mkl_dfti.h"
#include "CrysFFTPmmmBase.h"

/**
 * @class MklCrysFFTPmmm
 * @brief MKL-based crystallographic FFT using DCT via FFT.
 *
 * Implements DCT-II/III using 1D FFT per dimension with twiddle factor
 * preprocessing/postprocessing for O(N log N) complexity.
 */
class MklCrysFFTPmmm : public CrysFFTPmmmBase<MklCrysFFTPmmm>
{
private:
    using Base = CrysFFTPmmmBase<MklCrysFFTPmmm>;
    friend Base;

    // MKL DFTI descriptors for DCT via FFT (per dimension)
    std::array<DFTI_DESCRIPTOR_HANDLE, 3> dct_fft_forward_;
    std::array<DFTI_DESCRIPTOR_HANDLE, 3> dct_fft_backward_;

    // Precomputed twiddle factors for O(N log N) DCT via FFT
    std::array<std::vector<double>, 3> cos_tables_;
    std::array<std::vector<double>, 3> sin_tables_;

    /**
     * @brief Thread-local buffer structure.
     */
    struct ThreadBuffers {
        std::vector<double> io;
        std::vector<double> temp;
        int size = 0;
    };

    /**
     * @brief Get thread-local buffers.
     */
    std::pair<double*, double*> get_thread_local_buffers()
    {
        thread_local ThreadBuffers buffers;
        if (buffers.size != M_physical_)
        {
            buffers.io.resize(M_physical_);
            buffers.temp.resize(M_physical_);
            buffers.size = M_physical_;
        }
        return {buffers.io.data(), buffers.temp.data()};
    }

    /**
     * @brief Apply DCT-II forward via FFT.
     */
    void dct_forward_impl(double* io, double* temp);

    /**
     * @brief Apply DCT-III backward via FFT.
     */
    void dct_backward_impl(double* io, double* temp);

    /**
     * @brief Initialize MKL DFTI descriptors.
     */
    void initFFTPlans();

    /**
     * @brief Precompute twiddle factors.
     */
    void precomputeTwiddleFactors();

public:
    /**
     * @brief Construct MklCrysFFTPmmm for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     * @param trans_part (Ignored, kept for API compatibility)
     */
    MklCrysFFTPmmm(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    /**
     * @brief Destructor. Frees MKL descriptors and buffers.
     */
    ~MklCrysFFTPmmm();
};

#endif  // MKL_CRYS_FFT_PMMM_H_
