#ifndef CRYSFFT_SELECTOR_H_
#define CRYSFFT_SELECTOR_H_

#include <array>
#include <cmath>
#include "SpaceGroup.h"

enum class CrysFFTChoice
{
    None,
    Recursive3m,
    PmmmDct,
    ObliqueZ
};

struct CrysFFTSelection
{
    CrysFFTChoice mode = CrysFFTChoice::None;
    bool can_pmmm = false;
    std::array<double, 9> m3_translations = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool can_oblique_z = false;
    int oblique_z_shift = 0;
    double oblique_z_translation = 0.0;
};

inline CrysFFTSelection select_crysfft_mode(
    const SpaceGroup* sg,
    const std::array<int, 3>& nx,
    int dim,
    bool is_periodic,
    bool is_orthogonal,
    bool z_axis_orthogonal)
{
    CrysFFTSelection selection;

    if (sg == nullptr || dim != 3 || !is_periodic)
        return selection;

    const bool even_grid = (nx[0] % 2 == 0 && nx[1] % 2 == 0 && nx[2] % 2 == 0);
    const bool even_z = (nx[2] % 2 == 0);

    std::array<double, 9> trans_part = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const bool has_3m = sg->get_m3_translations(trans_part);
    const bool has_pmmm = sg->has_mirror_planes_xyz();
    // Recursive3m grid condition. The historical ((nx[2]/2) % 8 == 0) check was
    // neither necessary nor sufficient:
    // - Not necessary: equal grids work for ANY even nx with nx/2 >= 8
    //   (verified Im-3m/Fddd, space-group ON vs OFF at ~1e-13, for nx =
    //   16..44 including nx/2 = 10, 11, 12, 14, 18, 20, 21, 22 and 84^3).
    //   nx/2 < 8 must be rejected: the k-space multiply loops pad the packed
    //   z count to align_up(Nz2/2+1, 8) and overrun the buffer (heap
    //   corruption observed for nx = 12, 14).
    // - Not sufficient: some UNEQUAL grids are miscomputed even when
    //   (nz/2) % 8 == 0, e.g. Fddd on 84x48x16 / 84x48x32 (halves 42,24,8|16)
    //   gives wrong partition functions (~0.6 relative) on both MKL and FFTW,
    //   while 84x48x84 and 84x84x84 are fine — the failure correlates with
    //   mismatched quarter-parity between the x and z halves (42/2 = 21 odd
    //   vs 8/2 = 4 even). This is a latent bug in the shared 3m fold for
    //   unequal grids; until it is root-caused, unequal grids are allowed
    //   only when every half-extent is a multiple of 4 (all quarters even).
    //   This excludes every known-broken case. It is deliberately
    //   conservative: 84x48x84 (quarter-parity-matched, empirically fine)
    //   is also excluded and falls back to the standard path; a
    //   parity-match rule could admit it once the fold is root-caused.
    const bool even_all = (nx[0] % 2 == 0) && (nx[1] % 2 == 0) && (nx[2] % 2 == 0);
    const bool equal_grid = (nx[0] == nx[1]) && (nx[1] == nx[2]);
    const bool quarters_even = ((nx[0] / 2) % 4 == 0) &&
                               ((nx[1] / 2) % 4 == 0) &&
                               ((nx[2] / 2) % 4 == 0);
    const bool recursive_ok = even_all && ((nx[2] / 2) >= 8) &&
                              (equal_grid || quarters_even);

    selection.can_pmmm = has_pmmm;

    // Z-mirror (ObliqueZ): allow even for non-orthogonal cells when z-axis is orthogonal
    if (even_z && z_axis_orthogonal)
    {
        double tz = 0.0;
        const bool has_z_mirror = sg->get_z_mirror_translation(tz);
        if (has_z_mirror)
        {
            double t = tz;
            const double tol = 1e-8;
            if (std::fabs(t - 1.0) < tol)
                t = 0.0;
            if (std::fabs(t) < tol)
            {
                selection.can_oblique_z = true;
                selection.oblique_z_translation = t;
                selection.oblique_z_shift = 0;
            }
            else if (std::fabs(t - 0.5) < tol)
            {
                if ((nx[2] % 4) == 0)
                {
                    selection.can_oblique_z = true;
                    selection.oblique_z_translation = t;
                    selection.oblique_z_shift = nx[2] / 4;
                }
            }
        }
    }

    if (sg->using_z_mirror_physical_basis())
    {
        if (selection.can_oblique_z)
            selection.mode = CrysFFTChoice::ObliqueZ;
        return selection;
    }

    if (sg->using_pmmm_physical_basis())
    {
        if (has_pmmm)
            selection.mode = CrysFFTChoice::PmmmDct;
        return selection;
    }

    if (sg->using_m3_physical_basis())
    {
        if (has_3m && recursive_ok)
        {
            selection.mode = CrysFFTChoice::Recursive3m;
            selection.m3_translations = trans_part;
        }
        return selection;
    }

    if (!is_orthogonal)
    {
        selection.can_pmmm = false;
        return selection;
    }

    if (!even_grid)
        return selection;

    if (has_3m && recursive_ok)
    {
        selection.mode = CrysFFTChoice::Recursive3m;
        selection.m3_translations = trans_part;
        return selection;
    }

    if (has_pmmm)
        selection.mode = CrysFFTChoice::PmmmDct;

    if (selection.mode == CrysFFTChoice::None && selection.can_oblique_z)
        selection.mode = CrysFFTChoice::ObliqueZ;

    return selection;
}

#endif  // CRYSFFT_SELECTOR_H_
