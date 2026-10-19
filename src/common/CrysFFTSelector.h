#ifndef CRYSFFT_SELECTOR_H_
#define CRYSFFT_SELECTOR_H_

#include <array>
#include <cmath>
#include <vector>
#include "Recursive3mFoldParity.h"
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

/// Grid feasibility of the recursive 3m CrysFFT, and therefore of the m3
/// physical basis that relies on it: even grid, nz/2 >= 8, and a translation-
/// parity permutation that makes psi bijective (compute_m3_fold_permutation).
/// The proof that the psi-bijectivity test is equivalent to m3-basis
/// feasibility, and the grids it was verified on, are in
/// docs/theory/SpaceGroup.md (section "Recursive 3m fold parity").
/// On success trans_part holds the translations the engines must use (they
/// may be composed with a centering translation, see below).
inline bool crysfft_recursive_3m_feasible(
    const SpaceGroup* sg,
    const std::array<int, 3>& nx,
    std::array<double, 9>& trans_part)
{
    trans_part = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (!sg->get_m3_translations(trans_part))
        return false;
    const bool even_all = (nx[0] % 2 == 0) && (nx[1] % 2 == 0) && (nx[2] % 2 == 0);
    if (!even_all || (nx[2] / 2) < 8)
        return false;
    std::array<int, 8> fold_perm{};
    bool fold_ok = compute_m3_fold_permutation(trans_part, nx, fold_perm);
    if (!fold_ok)
    {
        // Minimal-norm representatives give a singular psi. Search coset
        // representatives composed with the pure (centering) translations;
        // by the exchange argument above, if the m3 basis is feasible at all,
        // some composition yields a bijective psi. The engines recompute the
        // permutation from selection.m3_translations, so storing the composed
        // translations keeps the fold, its twiddles, and psi consistent.
        std::vector<std::array<double, 3>> centerings;
        sg->get_pure_translations(centerings);
        auto wrap01 = [](double v) {
            v -= std::floor(v);
            if (v < 0.0)
                v += 1.0;
            return v;
        };
        for (size_t cz = 0; cz < centerings.size() && !fold_ok; ++cz)
        for (size_t cy = 0; cy < centerings.size() && !fold_ok; ++cy)
        for (size_t cx = 0; cx < centerings.size() && !fold_ok; ++cx)
        {
            std::array<double, 9> composed = trans_part;
            for (int d = 0; d < 3; ++d)
            {
                composed[0 + d] = wrap01(trans_part[0 + d] + centerings[cz][d]);
                composed[3 + d] = wrap01(trans_part[3 + d] + centerings[cy][d]);
                composed[6 + d] = wrap01(trans_part[6 + d] + centerings[cx][d]);
            }
            if (compute_m3_fold_permutation(composed, nx, fold_perm))
            {
                trans_part = composed;
                fold_ok = true;
            }
        }
    }
    return fold_ok;
}

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
    const bool recursive_ok = crysfft_recursive_3m_feasible(sg, nx, trans_part);
    const bool has_pmmm = sg->has_mirror_planes_xyz();

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
        if (recursive_ok)
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

    if (recursive_ok)
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
