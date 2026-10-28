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
#include <cmath>
#include <memory>

#include "CrysFFTSharedTables.h"
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
 * - Boltzmann factors shared by the calling threads (CrysFFTSharedTables.h)
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

    /// Boltzmann factors per ds, shared by the threads that call the engine
    SharedTableCache<double, std::unique_ptr<double[]>> boltzmann_;

    /// Boltzmann factors of the calling thread's last set_contour_step(); nullptr
    /// when it has called none since the last cell change.
    const double* current_boltzmann() const
    {
        const std::unique_ptr<double[]>* boltz = boltzmann_.current();
        return boltz ? boltz->get() : nullptr;
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
        boltzmann_.invalidate();
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
        boltzmann_.select(ds, [&] { return generateBoltzmann(ds); });
    }

    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
    int get_M_complex() const { return M_complex_; }
};

#endif  // CRYS_FFT_OBLIQUEZ_BASE_H_
