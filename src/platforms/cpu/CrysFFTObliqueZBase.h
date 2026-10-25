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
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>

#include "Exception.h"
#include "MinimalFormAlias.h"
#include "ReciprocalMetric.h"

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
 * - Per-thread state (owned by the object)
 *
 * The derived classes (FftwCrysFFTObliqueZ, MklCrysFFTObliqueZ) provide the
 * transforms: initFFTPlans(), diffusion(), apply_multiplier().
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

    struct ThreadState
    {
        std::map<double, std::unique_ptr<double[]>> boltzmann;
        const double* boltz_current = nullptr;
        double ds_current = std::numeric_limits<double>::quiet_NaN();
        uint64_t epoch = 0;
    };
    mutable std::atomic<uint64_t> cache_epoch_{1};   ///< bumped to invalidate every thread's cache
    /// Unique per object, never reused: lets a thread recognise the object its
    /// remembered cache slot belongs to (see get_thread_state()).
    inline static std::atomic<uint64_t> next_instance_id_{1};
    const uint64_t instance_id_{next_instance_id_.fetch_add(1, std::memory_order_relaxed)};
    mutable std::mutex thread_states_mutex_;
    /// Per-calling-thread caches (the propagator loop runs in OpenMP threads),
    /// owned by this object so that they are released together with it.
    mutable std::unordered_map<std::thread::id, ThreadState> thread_states_;

    /// Cache of the calling thread (owned by this object, see thread_states_).
    ThreadState& get_thread_state() const
    {
        // Each thread remembers the slot it used last, tagged with the object's
        // id, so the usual case (one object per thread) takes no lock. The slot
        // is looked up under the lock only when the thread switches object; the
        // map keeps references valid across insertions, and a stale remembered
        // pointer is never followed because ids are not reused.
        struct LastSlot { uint64_t instance_id = 0; ThreadState* state = nullptr; };
        thread_local LastSlot last;
        if (last.instance_id != instance_id_)
        {
            std::lock_guard<std::mutex> lock(thread_states_mutex_);
            last.state = &thread_states_[std::this_thread::get_id()];
            last.instance_id = instance_id_;
        }
        ThreadState* state = last.state;
        const uint64_t epoch = cache_epoch_.load(std::memory_order_acquire);
        if (state->epoch != epoch)
        {
            state->boltzmann.clear();
            state->boltz_current = nullptr;
            state->ds_current = std::numeric_limits<double>::quiet_NaN();
            state->epoch = epoch;
        }
        return *state;
    }

    std::unique_ptr<double[]> generateBoltzmann(double ds) const
    {
        auto boltz = std::make_unique<double[]>(M_complex_);

        const double G11 = recip_metric_[0];
        const double G12 = recip_metric_[1];
        const double G13 = recip_metric_[2];
        const double G22 = recip_metric_[3];
        const double G23 = recip_metric_[4];
        const double G33 = recip_metric_[5];
        const double factor = 4.0 * M_PI * M_PI;

        // For oblique in-plane cells (gamma != 90), pick the in-plane alias
        // representative minimizing the metric form, matching the invariant
        // (first-Brillouin-zone) convention used by Pseudo for the standard
        // full-grid path. This keeps the ObliqueZ diffusion operator exactly
        // symmetric under the hexagonal point group and mode-by-mode
        // consistent with the full-grid multiplier. The z (DCT) index is a
        // true non-negative frequency and needs no re-aliasing (ObliqueZ
        // requires alpha = beta = 90, i.e. G13 = G23 = 0).
        // Relative-tolerance gate: gamma of exactly 90 degrees gives
        // cos(pi/2) ~ 6e-17; do not flip rectangular cells onto the
        // minimal-form alias path.
        const bool oblique_xy = (std::abs(G12) > 1e-12 * std::max(std::abs(G11), std::abs(G22)));
        const std::array<int, 3> grid_xy = {nx_logical_[0], nx_logical_[1], 1};
        const IndexMetric metric_xy{G11, G22, 0.0, G12, 0.0, 0.0};
        auto form_xy = [&](double a, double b, double) {
            return G11 * a * a + 2.0 * G12 * a * b + G22 * b * b;
        };

        const int Nyh = nx_logical_[1] / 2 + 1;
        const int Nz2 = nx_physical_[2];
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            int m1s = (ix > nx_logical_[0] / 2) ? (ix - nx_logical_[0]) : ix;
            for (int iy = 0; iy < Nyh; ++iy)
            {
                std::array<int, 3> rep = {m1s, iy, 0};
                if (oblique_xy)
                    min_form_alias_rep(rep, grid_xy, metric_xy, form_xy);
                const int m1 = rep[0], m2 = rep[1];
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
          recip_metric_(reciprocal_metric_of_cell(cell_para))
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
        recip_metric_ = reciprocal_metric_of_cell(cell_para_);
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
            std::unique_ptr<double[]> boltz = generateBoltzmann(ds);
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
