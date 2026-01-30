/**
 * @file FftwCrysFFTHex.h
 * @brief CPU Crystallographic FFT for hexagonal z-mirror symmetry.
 *
 * Implements a minimal CrysFFT for hexagonal space groups with a z-mirror
 * operation (P6/mmm, P6_3/mmc). The transform uses:
 *   - DCT-II/III along z (half-grid)
 *   - Complex FFT along x and y (full grid)
 *
 * Logical grid: Nx × Ny × Nz (Nz even)
 * Physical grid: Nx × Ny × (Nz/2)
 *
 * The diffusion operator is applied in spectral space using the reciprocal
 * metric tensor from the cell parameters, supporting non-orthogonal cells.
 */

#ifndef FFTW_CRYS_FFT_HEX_H_
#define FFTW_CRYS_FFT_HEX_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include <fftw3.h>

class FftwCrysFFTHex
{
private:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_complex_{0};

    std::array<double, 6> cell_para_;
    std::array<double, 6> recip_metric_;

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

    fftw_plan plan_dct_forward_z_{nullptr};
    fftw_plan plan_dct_backward_z_{nullptr};
    fftw_plan plan_fft_forward_xy_{nullptr};
    fftw_plan plan_fft_backward_xy_{nullptr};
    fftw_plan plan_fft_z_forward_{nullptr};
    fftw_plan plan_fft_z_backward_{nullptr};

    double* io_buffer_{nullptr};
    double* temp_buffer_{nullptr};
    fftw_complex* complex_buffer_{nullptr};
    double* fft_z_real_{nullptr};
    fftw_complex* fft_z_complex_{nullptr};

    double norm_factor_{1.0};
    bool use_fft_dct_{false};
    int M_complex_z_{0};
    double fft_dct_scale_fwd_{1.0};
    double fft_dct_scale_bwd_{1.0};
    std::vector<double> dct_fft_cos_;
    std::vector<double> dct_fft_sin_;

    static std::array<double, 6> compute_recip_metric(const std::array<double, 6>& cell_para);
    double* generateBoltzmann(double ds) const;
    void initFFTPlans();
    void initFFTPlansZ(unsigned plan_flags);
    void calibrate_fft_dct_scale();
    void freeBoltzmann();
    ThreadState& get_thread_state() const;

public:
    FftwCrysFFTHex(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    ~FftwCrysFFTHex();

    void set_cell_para(const std::array<double, 6>& cell_para);
    void set_contour_step(double ds);

    void diffusion(double* q_in, double* q_out);
    void apply_multiplier(const double* q_in, double* q_out, const double* multiplier);

    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
};

#endif  // FFTW_CRYS_FFT_HEX_H_
