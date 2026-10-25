/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

/**
 * @file TestCrysFFTSelector.cpp
 * @brief The CrysFFT that select_crysfft_mode chooses (CrysFFTSelector.h).
 *
 * The space-group tests compare CrysFFT runs with full-grid runs, which agree
 * whatever CrysFFT is chosen; this test pins the choice itself:
 * - Oblique cells with alpha = beta = 90 degrees take ObliqueZ on the
 *   irreducible basis (P2/m, P6/mmm), with the glide shift nz/4 for P6_3/mmc;
 *   a z mirror at z = 1/4 needs nz divisible by 4.
 * - alpha != 90 degrees, non-periodic boxes, 2D grids and no space group give
 *   no CrysFFT.
 * - Orthogonal cells: Recursive3m for Im-3m and Pmmm when nz/2 >= 8 (the m3
 *   fold), the Pmmm DCT below that.
 * - Physical bases enabled by hand must match the cell.
 */

#include <array>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "CrysFFTSelector.h"
#include "SpaceGroup.h"

static bool check(bool condition, const std::string& what)
{
    std::cout << (condition ? "  ok: " : "  FAILED: ") << what << std::endl;
    return condition;
}

static bool throws(const std::function<void()>& call)
{
    try
    {
        call();
    }
    catch (std::exception&)
    {
        return true;
    }
    return false;
}

static const char* name(CrysFFTChoice mode)
{
    switch (mode)
    {
    case CrysFFTChoice::None: return "None";
    case CrysFFTChoice::Recursive3m: return "Recursive3m";
    case CrysFFTChoice::PmmmDct: return "PmmmDct";
    case CrysFFTChoice::ObliqueZ: return "ObliqueZ";
    }
    return "?";
}

// select_crysfft_mode for a 3D periodic box; orthogonal: all angles 90 degrees,
// z_orthogonal: alpha = beta = 90 degrees
static CrysFFTSelection select(const SpaceGroup& sg, std::array<int, 3> nx, bool orthogonal, bool z_orthogonal)
{
    return select_crysfft_mode(&sg, nx, 3, true, orthogonal, z_orthogonal);
}

static bool expect(const CrysFFTSelection& selection, CrysFFTChoice mode, const std::string& what)
{
    return check(selection.mode == mode, what + ": " + name(selection.mode) + " (expected " + name(mode) + ")");
}

int main()
{
    bool ok = true;
    try
    {
        std::cout << "=== Oblique cells, irreducible basis ===" << std::endl;
        {
            // Monoclinic, unique axis c: the mirror (x, y, -z), gamma = 100 degrees
            SpaceGroup p2m({11, 10, 8}, "P2/m", 58);
            const auto oblique = select(p2m, {11, 10, 8}, false, true);
            ok &= expect(oblique, CrysFFTChoice::ObliqueZ, "P2/m 11x10x8, gamma != 90");
            ok &= check(oblique.can_oblique_z && oblique.oblique_z_shift == 0 && !oblique.can_pmmm,
                "P2/m: z mirror at z = 0, no Pmmm DCT on an oblique cell");
            ok &= expect(select(p2m, {11, 10, 8}, false, false), CrysFFTChoice::None, "P2/m with alpha != 90");
        }
        {
            SpaceGroup p6mmm({12, 12, 8}, "P6/mmm");
            const auto hexagonal = select(p6mmm, {12, 12, 8}, false, true);
            ok &= expect(hexagonal, CrysFFTChoice::ObliqueZ, "P6/mmm 12x12x8, gamma = 120");
            ok &= check(hexagonal.oblique_z_shift == 0, "P6/mmm: z mirror at z = 0");
        }
        {
            // The z mirror of P6_3/mmc lies at z = 1/4: shift nz/4 on the grid
            SpaceGroup p63mmc({12, 12, 8}, "P6_3/mmc", 488);
            const auto glide = select(p63mmc, {12, 12, 8}, false, true);
            ok &= expect(glide, CrysFFTChoice::ObliqueZ, "P6_3/mmc 12x12x8");
            ok &= check(glide.oblique_z_shift == 2 && std::abs(glide.oblique_z_translation - 0.5) < 1e-12,
                "P6_3/mmc: shift nz/4 = 2 for the translation 1/2");
            SpaceGroup p63mmc_10({12, 12, 10}, "P6_3/mmc", 488);
            const auto odd_quarter = select(p63mmc_10, {12, 12, 10}, false, true);
            ok &= expect(odd_quarter, CrysFFTChoice::None, "P6_3/mmc 12x12x10 (nz/4 not an integer)");
            ok &= check(!odd_quarter.can_oblique_z, "P6_3/mmc 12x12x10: no ObliqueZ");
        }

        std::cout << "=== Orthogonal cells ===" << std::endl;
        {
            SpaceGroup im3m({16, 16, 16}, "Im-3m", 529);
            ok &= expect(select(im3m, {16, 16, 16}, true, true), CrysFFTChoice::Recursive3m, "Im-3m 16^3 (nz/2 = 8)");
            SpaceGroup pmmm({32, 24, 16}, "Pmmm");
            ok &= expect(select(pmmm, {32, 24, 16}, true, true), CrysFFTChoice::Recursive3m, "Pmmm 32x24x16 (nz/2 = 8)");
            SpaceGroup pmmm_small({12, 12, 10}, "Pmmm");
            const auto dct = select(pmmm_small, {12, 12, 10}, true, true);
            ok &= expect(dct, CrysFFTChoice::PmmmDct, "Pmmm 12x12x10 (nz/2 = 5, below the m3 fold)");
            ok &= check(dct.can_pmmm, "Pmmm 12x12x10: Pmmm DCT available");
        }

        std::cout << "=== No CrysFFT ===" << std::endl;
        {
            SpaceGroup p6mmm({12, 12, 8}, "P6/mmm");
            ok &= check(select_crysfft_mode(&p6mmm, {12, 12, 8}, 3, false, false, true).mode == CrysFFTChoice::None,
                "non-periodic box");
            ok &= check(select_crysfft_mode(&p6mmm, {12, 12, 8}, 2, true, false, true).mode == CrysFFTChoice::None,
                "2D grid");
            ok &= check(select_crysfft_mode(nullptr, {12, 12, 8}, 3, true, false, true).mode == CrysFFTChoice::None,
                "no space group");
        }

        std::cout << "=== Physical bases must match the cell ===" << std::endl;
        {
            SpaceGroup p6mmm({12, 12, 8}, "P6/mmm");
            p6mmm.enable_z_mirror_physical_basis();
            ok &= expect(select(p6mmm, {12, 12, 8}, false, true), CrysFFTChoice::ObliqueZ, "P6/mmm z-mirror basis");
            ok &= check(throws([&]{ select(p6mmm, {12, 12, 8}, false, false); }),
                "the z-mirror basis rejects alpha != 90");
            SpaceGroup pmmm({12, 12, 10}, "Pmmm");
            pmmm.enable_pmmm_physical_basis();
            ok &= expect(select(pmmm, {12, 12, 10}, true, true), CrysFFTChoice::PmmmDct, "Pmmm basis");
            ok &= check(throws([&]{ select(pmmm, {12, 12, 10}, false, true); }),
                "the Pmmm basis rejects an oblique cell");
        }
    }
    catch (std::exception& exc)
    {
        std::cout << "Unexpected exception: " << exc.what() << std::endl;
        return -1;
    }

    if (!ok)
    {
        std::cout << "TestCrysFFTSelector FAILED" << std::endl;
        return -1;
    }
    std::cout << "TestCrysFFTSelector passed" << std::endl;
    return 0;
}
