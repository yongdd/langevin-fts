/**
 * @file MklCrysFFTRecursive3m.h
 * @brief MKL recursive crystallographic FFT (2x2y2z) for 3 perpendicular mirrors.
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
 * - MKL implementation using DFTI for FFT operations.
 * - Thread safety: internal DFTI descriptors are shared, but all input/output
 *   buffers are thread-local to allow concurrent calls.
 */

#ifndef MKL_CRYS_FFT_RECURSIVE_3M_H_
#define MKL_CRYS_FFT_RECURSIVE_3M_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <complex>
#include "mkl_dfti.h"

class MklCrysFFTRecursive3m
{
public:
    enum class MultiplierType
    {
        Kx2,
        Ky2,
        Kz2,
        ExpKx2,
        ExpKy2,
        ExpKz2
    };

    MklCrysFFTRecursive3m(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> translational_part = {0,0,0, 0,0,0, 0,0,0});

    ~MklCrysFFTRecursive3m();

    void set_cell_para(const std::array<double, 6>& cell_para);
    void set_contour_step(double coeff);

    void diffusion(const double* q_in, double* q_out) const;

    void apply_multiplier(const double* q_in, double* q_out, MultiplierType type, double coeff) const;

    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }

private:
    struct AlignedDeleter {
        void operator()(double* ptr) const { if (ptr) delete[] ptr; }
        void operator()(std::complex<double>* ptr) const { if (ptr) delete[] ptr; }
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

    struct ThreadState {
        std::map<double, KCache> k_cache;
        const KCache* k_current = nullptr;
        double coeff_current = std::numeric_limits<double>::quiet_NaN();
        std::map<double, KCache> exp_kx2_cache;
        std::map<double, KCache> exp_ky2_cache;
        std::map<double, KCache> exp_kz2_cache;
        std::unique_ptr<KCache> kx2_cache;
        std::unique_ptr<KCache> ky2_cache;
        std::unique_ptr<KCache> kz2_cache;
        uint64_t epoch = 0;
        uint64_t instance_id = 0;
    };
    inline static std::atomic<uint64_t> next_instance_id_{1};
    uint64_t instance_id_{0};
    mutable std::atomic<uint64_t> cache_epoch_{1};

    DFTI_DESCRIPTOR_HANDLE plan_forward_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE plan_backward_ = nullptr;

    void init_fft_plans();
    void generate_twiddle_factors();
    KCache generate_k_cache(double coeff) const;
    ThreadState& get_thread_state() const;
    KCache generate_k_cache_from_multiplier(MultiplierType type, double coeff) const;
    const KCache& get_multiplier_cache(MultiplierType type, double coeff) const;
    void apply_with_cache(const KCache& cache, const double* q_in, double* q_out) const;
};

#endif  // MKL_CRYS_FFT_RECURSIVE_3M_H_
