/**
 * @file CrysFFTObliqueZBase.h
 * @brief CRTP base class for crystallographic FFT with z-mirror symmetry.
 *
 * Implements the ObliqueZ algorithm for space groups with a z-mirror
 * operation (t_z = 0 or 1/2) and alpha=beta=90° (gamma arbitrary).
 *
 * @see FftwCrysFFTObliqueZ for FFTW implementation
 * @see MklCrysFFTObliqueZ for MKL implementation
 */

#ifndef CRYS_FFT_OBLIQUEZ_BASE_H_
#define CRYS_FFT_OBLIQUEZ_BASE_H_

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>

#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class CrysFFTObliqueZBase
 * @brief CRTP base class for ObliqueZ crystallographic FFT.
 *
 * Provides common functionality:
 * - Reciprocal metric tensor computation
 * - Boltzmann factor generation and caching
 * - Thread-local state management
 *
 * Derived classes must implement:
 * - initFFTPlans(): Create platform-specific FFT/DCT plans
 * - diffusion(): Apply diffusion operator
 * - apply_multiplier(): Apply arbitrary multiplier in spectral space
 *
 * @tparam Derived The derived class type (CRTP pattern)
 */
template <typename Derived>
class CrysFFTObliqueZBase
{
protected:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_complex_{0};

    std::array<double, 6> cell_para_;
    std::array<double, 6> recip_metric_;

    double norm_factor_{1.0};

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

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    static std::array<double, 6> compute_recip_metric(const std::array<double, 6>& cell_para)
    {
        const double a = cell_para[0];
        const double b = cell_para[1];
        const double c = cell_para[2];
        const double alpha = cell_para[3];
        const double beta = cell_para[4];
        const double gamma = cell_para[5];

        const double cos_a = std::cos(alpha);
        const double cos_b = std::cos(beta);
        const double cos_g = std::cos(gamma);
        const double sin_g = std::sin(gamma);

        const double vol_factor_sq =
            1.0 - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g + 2.0 * cos_a * cos_b * cos_g;
        const double vol_factor = std::sqrt(vol_factor_sq);
        const double volume = a * b * c * vol_factor;

        // Direct lattice vectors
        const double ax = a, ay = 0.0, az = 0.0;
        const double bx = b * cos_g, by = b * sin_g, bz = 0.0;
        const double cx = c * cos_b;
        const double cy = (sin_g != 0.0) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
        const double cz = volume / (a * b * sin_g);

        // Reciprocal lattice vectors (without 2π)
        const double bc_x = by * cz - bz * cy;
        const double bc_y = bz * cx - bx * cz;
        const double bc_z = bx * cy - by * cx;

        const double ca_x = cy * az - cz * ay;
        const double ca_y = cz * ax - cx * az;
        const double ca_z = cx * ay - cy * ax;

        const double ab_x = ay * bz - az * by;
        const double ab_y = az * bx - ax * bz;
        const double ab_z = ax * by - ay * bx;

        const double a_star_x = bc_x / volume;
        const double a_star_y = bc_y / volume;
        const double a_star_z = bc_z / volume;
        const double b_star_x = ca_x / volume;
        const double b_star_y = ca_y / volume;
        const double b_star_z = ca_z / volume;
        const double c_star_x = ab_x / volume;
        const double c_star_y = ab_y / volume;
        const double c_star_z = ab_z / volume;

        std::array<double, 6> metric{};
        metric[0] = a_star_x * a_star_x + a_star_y * a_star_y + a_star_z * a_star_z;
        metric[1] = a_star_x * b_star_x + a_star_y * b_star_y + a_star_z * b_star_z;
        metric[2] = a_star_x * c_star_x + a_star_y * c_star_y + a_star_z * c_star_z;
        metric[3] = b_star_x * b_star_x + b_star_y * b_star_y + b_star_z * b_star_z;
        metric[4] = b_star_x * c_star_x + b_star_y * c_star_y + b_star_z * c_star_z;
        metric[5] = c_star_x * c_star_x + c_star_y * c_star_y + c_star_z * c_star_z;
        return metric;
    }

    ThreadState& get_thread_state() const
    {
        struct ThreadLocalStates
        {
            std::unordered_map<const CrysFFTObliqueZBase*, ThreadState> states;
        };
        thread_local ThreadLocalStates tls;

        ThreadState& state = tls.states[this];
        const uint64_t epoch = cache_epoch_.load(std::memory_order_acquire);
        if (state.instance_id != instance_id_ || state.epoch != epoch)
        {
            state.boltzmann.clear();
            state.boltz_current = nullptr;
            state.ds_current = std::numeric_limits<double>::quiet_NaN();
            state.epoch = epoch;
            state.instance_id = instance_id_;
        }
        return state;
    }

    double* generateBoltzmann(double ds) const
    {
        double* boltz = new double[M_complex_];

        const double G11 = recip_metric_[0];
        const double G12 = recip_metric_[1];
        const double G13 = recip_metric_[2];
        const double G22 = recip_metric_[3];
        const double G23 = recip_metric_[4];
        const double G33 = recip_metric_[5];
        const double factor = 4.0 * M_PI * M_PI;

        const int Nyh = nx_logical_[1] / 2 + 1;
        const int Nz2 = nx_physical_[2];
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            int m1 = (ix > nx_logical_[0] / 2) ? (ix - nx_logical_[0]) : ix;
            for (int iy = 0; iy < Nyh; ++iy)
            {
                int m2 = iy;
                for (int iz = 0; iz < Nz2; ++iz)
                {
                    int m3 = iz;
                    double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                               + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                    double k2 = factor * gmm;
                    int idx = (ix * Nyh + iy) * Nz2 + iz;
                    boltz[idx] = std::exp(-k2 * ds);
                }
            }
        }

        return boltz;
    }

    void freeBoltzmann()
    {
        cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
    }

public:
    CrysFFTObliqueZBase(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> /*trans_part*/)
        : nx_logical_(nx_logical),
          cell_para_(cell_para),
          recip_metric_(compute_recip_metric(cell_para)),
          instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed))
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] <= 0)
                throw_with_line_number("CrysFFTObliqueZ requires positive grid dimensions.");
        }
        if (nx_logical_[2] % 2 != 0)
            throw_with_line_number("CrysFFTObliqueZ requires even Nz.");

        nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
        M_complex_ = nx_logical_[0] * (nx_logical_[1] / 2 + 1) * nx_physical_[2];
        norm_factor_ = 1.0 / static_cast<double>(M_logical_);
    }

    virtual ~CrysFFTObliqueZBase() = default;

    void set_cell_para(const std::array<double, 6>& cell_para)
    {
        if (cell_para == cell_para_)
            return;
        cell_para_ = cell_para;
        recip_metric_ = compute_recip_metric(cell_para_);
        freeBoltzmann();
    }

    void set_contour_step(double ds)
    {
        ThreadState& state = get_thread_state();
        if (state.boltz_current != nullptr && state.ds_current == ds)
            return;

        auto it = state.boltzmann.find(ds);
        if (it == state.boltzmann.end())
        {
            std::unique_ptr<double, BoltzDeleter> boltz(generateBoltzmann(ds));
            it = state.boltzmann.emplace(ds, std::move(boltz)).first;
        }

        state.ds_current = ds;
        state.boltz_current = it->second.get();
    }

    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
    int get_M_complex() const { return M_complex_; }
};

#endif  // CRYS_FFT_OBLIQUEZ_BASE_H_
