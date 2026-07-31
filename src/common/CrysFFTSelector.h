#ifndef CRYSFFT_SELECTOR_H_
#define CRYSFFT_SELECTOR_H_

#include <array>
#include <cmath>
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
    // Recursive3m grid condition: even grid, nz/2 >= 8, and a valid
    // translation-parity permutation (compute_m3_fold_permutation). The fold
    // pairs twiddle r_p with the octant-transformed Boltzmann array S_{psi(p)}
    // where psi depends on the parities of the m3 generator translations in
    // grid units (tau = t*N); psi must be a bijection. Empirically this has
    // matched the feasibility of the even-index m3 physical basis in every
    // tested case (for Fddd's quarter-cell d-glides, at most one of Nx/4,
    // Ny/4, Nz/4 may be odd), but the equivalence is not proven in general:
    // psi uses the minimal-norm coset representative per generator, while
    // basis feasibility considers all centering-composed representatives.
    // Historically the code hard-coded psi = identity, which silently
    // miscomputed grids with odd tau (e.g. Fddd 84x48x16/24/32, wrong Q at
    // ~0.6 relative); an interim guard restricted unequal grids to
    // all-quarters-even. With the psi pairing implemented in
    // CrysFFTRecursive3mBase/CudaCrysFFTRecursive3m, verified 2026-08:
    // - Unit level (one diffusion vs full-grid FFT reference, <= 1e-14 rel),
    //   MKL and FFTW: Fddd 84x48x16, 84x48x24, 84x48x32 (previously broken),
    //   84x56x16, 84x56x32 (odd qx); 64x52x16, 72x52x16 (odd qy); 64x48x84,
    //   72x56x84, 64x48x20, 64x48x12 (odd qz); 64x48x16, 88x48x24, 64x56x32,
    //   32x24x16, 32x24x8 (all even); Fddd 16/24/40/48^3; Im-3m 16/20/22/
    //   40/44^3. CUDA: 84x48x16, 64x52x16, 64x48x84, 64x48x12, Im-3m 40^3,
    //   including the k^2-multiplier (stress) caches.
    // - SCFT space-group ON-vs-OFF partition comparisons (<= 7e-14 rel):
    //   the trio on cpu-mkl AND cpu-fftw; 64x48x16, 84x56x16, 64x52x16,
    //   64x48x84 on cpu-mkl; Fddd 84x48x16, 64x48x84 and Im-3m 40^3 on cuda.
    // - Grids whose psi is singular (verified Fddd 44^3, 84^3, 84x48x84 —
    //   two or more odd quarters) are rejected here and fall back to the
    //   standard FFT path (still correct, per the same ON-vs-OFF probes),
    //   matching SpaceGroup m3-basis feasibility exactly in all cases above.
    // nz/2 >= 8 is kept as the established performance/selection bound;
    // smaller nz still works via the standard path (the former buffer-overrun
    // reason is gone: the padded k-loop is now clamped to the row stride).
    const bool even_all = (nx[0] % 2 == 0) && (nx[1] % 2 == 0) && (nx[2] % 2 == 0);
    std::array<int, 8> fold_perm{};
    const bool fold_ok = has_3m &&
        compute_m3_fold_permutation(trans_part, nx, fold_perm);
    const bool recursive_ok = even_all && ((nx[2] / 2) >= 8) && fold_ok;

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
