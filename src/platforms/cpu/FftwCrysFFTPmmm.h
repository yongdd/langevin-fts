/**
 * @file FftwCrysFFTPmmm.h
 * @brief FFTW implementation of crystallographic FFT for Pmmm symmetry.
 *
 * Uses native FFTW3 DCT-II/III (REDFT10/REDFT01) for O(N log N) diffusion
 * computation with Pmmm mirror symmetry.
 *
 * @see CrysFFTPmmmBase for common functionality
 * @see MklCrysFFTPmmm for MKL implementation
 */

#ifndef FFTW_CRYS_FFT_PMMM_H_
#define FFTW_CRYS_FFT_PMMM_H_

#include <array>
#include <tuple>
#include <fftw3.h>
#include "CrysFFTPmmmBase.h"

/**
 * @class FftwCrysFFTPmmm
 * @brief FFTW-based crystallographic FFT using native DCT-II/III.
 *
 * Uses FFTW's real-to-real 3D DCT plans (REDFT10/REDFT01) for
 * efficient computation of Pmmm symmetric diffusion.
 */
class FftwCrysFFTPmmm : public CrysFFTPmmmBase<FftwCrysFFTPmmm>
{
private:
    using Base = CrysFFTPmmmBase<FftwCrysFFTPmmm>;
    friend Base;

    // FFTW plans for DCT-II/III
    fftw_plan plan_forward_;    ///< Forward DCT-II (REDFT10)
    fftw_plan plan_backward_;   ///< Backward DCT-III (REDFT01)

    /**
     * @brief Thread-local buffer structure for FFTW.
     */
    struct AlignedDeleter {
        void operator()(double* ptr) const { if (ptr) fftw_free(ptr); }
    };

    struct ThreadBuffers {
        std::unique_ptr<double, AlignedDeleter> io;
        std::unique_ptr<double, AlignedDeleter> temp;
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
            buffers.io.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
            buffers.temp.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
            if (!buffers.io || !buffers.temp)
            {
                throw_with_line_number("Failed to allocate thread-local FFTW buffers.");
            }
            buffers.size = M_physical_;
        }
        return {buffers.io.get(), buffers.temp.get()};
    }

    /**
     * @brief Apply DCT-II forward (FFTW REDFT10).
     */
    void dct_forward_impl(double* io, double* temp)
    {
        fftw_execute_r2r(plan_forward_, io, temp);
        std::memcpy(io, temp, sizeof(double) * M_physical_);
    }

    /**
     * @brief Apply DCT-III backward (FFTW REDFT01).
     */
    void dct_backward_impl(double* io, double* temp)
    {
        fftw_execute_r2r(plan_backward_, io, temp);
        std::memcpy(io, temp, sizeof(double) * M_physical_);
    }

public:
    /**
     * @brief Construct FftwCrysFFTPmmm for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     * @param trans_part (Ignored, kept for API compatibility)
     */
    FftwCrysFFTPmmm(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    /**
     * @brief Destructor. Frees FFTW plans and buffers.
     */
    ~FftwCrysFFTPmmm();
};

#endif  // FFTW_CRYS_FFT_PMMM_H_
