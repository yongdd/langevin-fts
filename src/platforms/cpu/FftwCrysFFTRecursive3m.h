/**
 * @file FftwCrysFFTRecursive3m.h
 * @brief CPU recursive crystallographic FFT (2x2y2z) for 3 perpendicular mirrors.
 *
 * Implements the recursive 2x2y2z algorithm described in
 * "Accelerated pseudo-spectral method of self-consistent field theory via
 * crystallographic fast Fourier transform", Macromolecules (2020).
 *
 * This class operates on the *physical grid* (Nx/2, Ny/2, Nz/2) corresponding
 * to Pmmm symmetry (three perpendicular mirror planes). It performs a
 * diffusion step:
 *   q_out = CrysFFT^{-1}[ exp(-k^2 * coeff) * CrysFFT[q_in] ]
 *
 * where coeff = (b^2 * ds / 6).
 *
 * Notes:
 * - Only CPU implementation is provided.
 * - Thread safety: internal FFTW plans are shared, but all input/output
 *   buffers are thread-local to allow concurrent calls.
 */

#ifndef FFTW_CRYS_FFT_RECURSIVE_3M_H_
#define FFTW_CRYS_FFT_RECURSIVE_3M_H_

#include <array>
#include <map>
#include <memory>
#include <vector>
#include <fftw3.h>

class FftwCrysFFTRecursive3m
{
public:
    FftwCrysFFTRecursive3m(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> translational_part = {0,0,0, 0,0,0, 0,0,0});

    ~FftwCrysFFTRecursive3m();

    void set_cell_para(const std::array<double, 6>& cell_para);
    void set_contour_step(double coeff);

    void diffusion(const double* q_in, double* q_out) const;

    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }

private:
    struct AlignedDeleter {
        void operator()(double* ptr) const { if (ptr) fftw_free(ptr); }
        void operator()(fftw_complex* ptr) const { if (ptr) fftw_free(ptr); }
    };

    struct KCache {
        std::array<std::unique_ptr<double, AlignedDeleter>, 8> re;
        std::array<std::unique_ptr<double, AlignedDeleter>, 8> im;
    };

    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_;
    int M_physical_;
    std::array<double, 6> cell_para_;
    std::array<double, 9> translational_part_;

    std::array<std::unique_ptr<double, AlignedDeleter>, 8> r_re_;
    std::array<std::unique_ptr<double, AlignedDeleter>, 8> r_im_;

    std::map<double, KCache> k_cache_;
    const KCache* k_current_ = nullptr;
    double coeff_current_ = 0.0;

    fftw_plan plan_forward_ = nullptr;
    fftw_plan plan_backward_ = nullptr;

    void init_fft_plans();
    void generate_twiddle_factors();
    void generate_k_cache(double coeff);

    // No public expansion helpers; mapping is handled by callers.
};

#endif  // FFTW_CRYS_FFT_RECURSIVE_3M_H_
