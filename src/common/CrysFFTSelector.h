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
    HexZ
};

struct CrysFFTSelection
{
    CrysFFTChoice mode = CrysFFTChoice::None;
    bool can_pmmm = false;
    std::array<double, 9> m3_translations = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool can_hex_z = false;
    int hex_z_shift = 0;
    double hex_z_translation = 0.0;
};

inline CrysFFTSelection select_crysfft_mode(
    const SpaceGroup* sg,
    const std::array<int, 3>& nx,
    int dim,
    bool is_periodic,
    bool is_orthogonal)
{
    CrysFFTSelection selection;

    if (sg == nullptr || dim != 3 || !is_periodic)
        return selection;

    const bool even_grid = (nx[0] % 2 == 0 && nx[1] % 2 == 0 && nx[2] % 2 == 0);
    const bool even_z = (nx[2] % 2 == 0);

    std::array<double, 9> trans_part = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const bool has_3m = sg->get_m3_translations(trans_part);
    const bool has_pmmm = sg->has_mirror_planes_xyz();
    const bool recursive_ok = ((nx[2] / 2) % 8) == 0;

    selection.can_pmmm = has_pmmm;

    // Hexagonal z-mirror (P6/mmm, P6_3/mmc): allow even for non-orthogonal cells
    if (even_z)
    {
        double tz = 0.0;
        const bool has_z_mirror = sg->get_z_mirror_translation(tz);
        const int sg_num = sg->get_spacegroup_number();
        const bool is_hex_target = (sg_num == 191 || sg_num == 194);
        if (has_z_mirror && is_hex_target)
        {
            double t = tz;
            const double tol = 1e-8;
            if (std::fabs(t - 1.0) < tol)
                t = 0.0;
            if (std::fabs(t) < tol)
            {
                selection.can_hex_z = true;
                selection.hex_z_translation = t;
                selection.hex_z_shift = 0;
            }
            else if (std::fabs(t - 0.5) < tol)
            {
                // Glide mirror (t_z = 1/2) requires a dedicated transform; disable for now.
                selection.can_hex_z = false;
            }
        }
    }

    if (sg->using_z_mirror_physical_basis())
    {
        if (selection.can_hex_z)
            selection.mode = CrysFFTChoice::HexZ;
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
        if (selection.can_hex_z)
            selection.mode = CrysFFTChoice::HexZ;
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

    if (selection.mode == CrysFFTChoice::None && selection.can_hex_z)
        selection.mode = CrysFFTChoice::HexZ;

    return selection;
}

#endif  // CRYSFFT_SELECTOR_H_
