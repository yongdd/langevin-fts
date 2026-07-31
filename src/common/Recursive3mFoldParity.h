/**
 * @file Recursive3mFoldParity.h
 * @brief Translation-parity pairing permutation for the recursive 3m CrysFFT.
 *
 * The 2x2y2z fold reconstructs the eight alias-spectrum values F(m + o*n)
 * (o in {0,1}^3, n = N/2 the half grid) of the full symmetric field from the
 * half-grid FFT values Qhat(D_a m) of the even-index decimated field, using
 * the mirror relations F(K) = c_a(K) F(D_a K) with
 * c_a(K) = exp(-2*pi*i K . t_eff_a).  Folding an alias offset o*n through
 * c_a produces the per-element sign
 *
 *     chi_a(o)^{-1} = (-1)^{psi_a . o},   psi_a[d] = (a[d] + tau_a[d]) mod 2,
 *
 * where tau_a[d] is element a's combined translation in grid units
 * (tau = t*N; the extra a[d] term comes from the cell-centered -1/N offset on
 * flipped axes).  Inverting the resulting Hadamard system shows that the
 * octant-transformed Boltzmann array S_p(m) = sum_o (-1)^{p.o} B(m + o*n)
 * paired with twiddle r_a must be S_{psi(a)}, not S_a.  The identity pairing
 * is correct only when every generator translation is an even number of grid
 * cells; e.g. Fddd's d-glides translate by a quarter cell (tau = N/4), so any
 * odd quarter makes psi != identity.
 *
 * psi is linear over GF(2); it is a bijection exactly when the m3 physical
 * basis (even-index subgrid) exists for this grid.  When psi is singular some
 * grid-point parity class contains no even-index representative, the basis
 * construction fails, and the fold cannot represent the group on this grid.
 */

#ifndef RECURSIVE_3M_FOLD_PARITY_H_
#define RECURSIVE_3M_FOLD_PARITY_H_

#include <array>
#include <cmath>

/**
 * @brief Compute the S<->r pairing permutation of the recursive 3m fold.
 *
 * @param g   The nine m3 generator translations as cell fractions, ordered as
 *            in SpaceGroup::get_m3_translations: m_z = g[0..2], m_y = g[3..5],
 *            m_x = g[6..8].
 * @param nx  Logical (full) grid dimensions.
 * @param perm Filled with perm[a] = psi(a) on success.  Octant bit code:
 *            bit 2 = x, bit 1 = y, bit 0 = z (the mat_split convention).
 * @return true when all generator translations are integers in grid units and
 *         psi is a bijection; false otherwise (grid unsupported by the fold).
 */
inline bool compute_m3_fold_permutation(
    const std::array<double, 9>& g,
    const std::array<int, 3>& nx,
    std::array<int, 8>& perm)
{
    // Parity code of each generator's translation vector in grid units.
    // parity_code[0]: m_z (element bit 0), [1]: m_y (bit 1), [2]: m_x (bit 2).
    int parity_code[3];
    for (int gen = 0; gen < 3; ++gen)
    {
        int code = 0;
        for (int d = 0; d < 3; ++d)
        {
            const double tau = g[3 * gen + d] * nx[d];
            const long long tau_round = std::llround(tau);
            if (std::fabs(tau - static_cast<double>(tau_round)) > 1e-8)
                return false;  // translation incommensurate with the grid
            const int par = static_cast<int>(tau_round & 1LL);
            code |= par << (2 - d);
        }
        parity_code[gen] = code;
    }
    const int p_mz = parity_code[0];
    const int p_my = parity_code[1];
    const int p_mx = parity_code[2];

    unsigned seen = 0;
    for (int a = 0; a < 8; ++a)
    {
        int v = a;
        if (a & 4) v ^= p_mx;
        if (a & 2) v ^= p_my;
        if (a & 1) v ^= p_mz;
        perm[a] = v;
        seen |= 1u << v;
    }
    return seen == 0xFFu;
}

#endif  // RECURSIVE_3M_FOLD_PARITY_H_
