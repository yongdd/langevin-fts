/**
 * @file MinimalFormAlias.h
 * @brief Exact search for the alias of a Fourier mode that minimizes the
 *        reciprocal-metric form (first-Brillouin-zone representative).
 *
 * On a periodic grid a mode m is defined modulo the grid sizes N, and |k|² must
 * use the alias u = m + (a N1, b N2, c N3) of least form f(u) = uᵀ G u, which
 * is invariant under the point group. A window of one period around the
 * centered representative finds it only when the lattice of the periods is
 * reduced in the metric. On oblique cells with strongly anisotropic grid
 * spacing (e.g. 4x32 points on a 120-degree cell with equal lengths) it misses
 * the minimum, and the factors of k and -k then differ. The form of the centered
 * representative bounds the search instead: every alias with a form not above
 * it satisfies |u_d| <= sqrt(f(m) (G⁻¹)_dd), so enumerating that box is exact.
 * Where the minimum lies within one period, the box is inside the old window
 * and the aliases are visited in the same (a, b, c) order, so the results do not
 * change there.
 *
 * Axes with N_d <= 1 or a zero diagonal metric (the padded axes of 1D and 2D
 * grids) are not searched.
 */

#ifndef MINIMAL_FORM_ALIAS_H_
#define MINIMAL_FORM_ALIAS_H_

#include <array>
#include <cmath>

/// Reciprocal metric in the index space of the search.
struct IndexMetric
{
    double g11, g22, g33, g12, g13, g23;
};

namespace min_form_alias_detail
{
/// Ranges of the period shifts (a, b, c) that can hold an alias whose form does
/// not exceed f_center, the form of m itself.
inline void search_box(
    const std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g,
    double f_center, std::array<int, 3>& lo, std::array<int, 3>& hi)
{
    const double G[3][3] = {{g.g11, g.g12, g.g13}, {g.g12, g.g22, g.g23}, {g.g13, g.g23, g.g33}};
    int active[3];
    int n_active = 0;
    for (int d = 0; d < 3; ++d)
    {
        lo[d] = hi[d] = 0;
        if (n[d] > 1 && G[d][d] > 0.0)
            active[n_active++] = d;
    }
    if (n_active == 0 || !(f_center > 0.0))
        return;

    // Diagonal of the inverse of the metric restricted to the searched axes
    double inv_diag[3] = {0.0, 0.0, 0.0};
    if (n_active == 1)
    {
        inv_diag[0] = 1.0 / G[active[0]][active[0]];
    }
    else if (n_active == 2)
    {
        const double p = G[active[0]][active[0]], q = G[active[0]][active[1]], r = G[active[1]][active[1]];
        const double det = p * r - q * q;
        inv_diag[0] = r / det;
        inv_diag[1] = p / det;
    }
    else
    {
        const double det = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[1][2])
                         - G[0][1] * (G[0][1] * G[2][2] - G[1][2] * G[0][2])
                         + G[0][2] * (G[0][1] * G[1][2] - G[1][1] * G[0][2]);
        inv_diag[0] = (G[1][1] * G[2][2] - G[1][2] * G[1][2]) / det;
        inv_diag[1] = (G[0][0] * G[2][2] - G[0][2] * G[0][2]) / det;
        inv_diag[2] = (G[0][0] * G[1][1] - G[0][1] * G[0][1]) / det;
    }

    for (int i = 0; i < n_active; ++i)
    {
        const int d = active[i];
        if (!(inv_diag[i] > 0.0) || !std::isfinite(inv_diag[i]))
        {
            // Degenerate metric: fall back to one period on each searched axis
            for (int j = 0; j < n_active; ++j)
            {
                lo[active[j]] = -1;
                hi[active[j]] = 1;
            }
            return;
        }
        // A small slack only adds candidates, so rounding cannot drop an alias
        const double reach = std::sqrt(f_center * inv_diag[i]) * (1.0 + 1e-9) + 1e-9;
        lo[d] = static_cast<int>(std::ceil((-reach - m[d]) / n[d]));
        hi[d] = static_cast<int>(std::floor((reach - m[d]) / n[d]));
        if (lo[d] > 0) lo[d] = 0;
        if (hi[d] < 0) hi[d] = 0;
    }
}
}  // namespace min_form_alias_detail

/**
 * @brief Replace m by its alias of least form.
 *
 * `form(u1, u2, u3)` evaluates the form; callers pass their own expression so
 * that its rounding is unchanged. The aliases are visited in (a, b, c) order and
 * one replaces the current choice only when its form is strictly smaller, so m
 * itself is kept on ties, as with the old one-period window.
 */
template <typename Form>
inline void min_form_alias_rep(
    std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g, Form&& form)
{
    const double f_center = form(m[0], m[1], m[2]);
    std::array<int, 3> lo, hi;
    min_form_alias_detail::search_box(m, n, g, f_center, lo, hi);

    std::array<int, 3> best_rep = m;
    double best = f_center;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
            {
                if (a == 0 && b == 0 && c == 0)
                    continue;
                const int t1 = m[0] + a * n[0];
                const int t2 = m[1] + b * n[1];
                const int t3 = m[2] + c * n[2];
                const double f = form(t1, t2, t3);
                if (f < best)
                {
                    best = f;
                    best_rep = {t1, t2, t3};
                }
            }
    m = best_rep;
}

/**
 * @brief Visit, in (a, b, c) order, every alias of m whose form lies within
 *        rel_tol * minimum of the minimum; returns the number visited.
 */
template <typename Form, typename Visit>
inline int for_each_min_form_alias(
    const std::array<int, 3>& m, const std::array<int, 3>& n, const IndexMetric& g,
    Form&& form, double rel_tol, Visit&& visit)
{
    const double f_center = form(m[0], m[1], m[2]);
    std::array<int, 3> lo, hi;
    min_form_alias_detail::search_box(m, n, g, f_center, lo, hi);

    double best = f_center;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
            {
                const double f = form(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]);
                if (f < best)
                    best = f;
            }

    const double tol = rel_tol * best;
    int count = 0;
    for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
            for (int c = lo[2]; c <= hi[2]; ++c)
            {
                const int t1 = m[0] + a * n[0];
                const int t2 = m[1] + b * n[1];
                const int t3 = m[2] + c * n[2];
                if (form(t1, t2, t3) > best + tol)
                    continue;
                visit(t1, t2, t3);
                ++count;
            }
    return count;
}

#endif  // MINIMAL_FORM_ALIAS_H_
