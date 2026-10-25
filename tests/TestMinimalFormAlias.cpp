/**
 * @file TestMinimalFormAlias.cpp
 * @brief The alias search of MinimalFormAlias.h against a brute-force search.
 *
 * For random oblique (2D) and triclinic (3D) metrics, including strongly
 * anisotropic grid spacing, every mode's alias of least form must match a wide
 * brute-force window, the least form must be the same for m and -m (which the
 * CUDA ObliqueZ spectral kernel needs), and for_each_min_form_alias must visit
 * the same number of tied aliases. On cells with nearly isotropic spacing the
 * representative must equal the one of the former one-period window, so that
 * results do not change there. The case that motivated the exact search, 4x32
 * points on a 120-degree cell with equal lengths, must defeat that window.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>

#include "MinimalFormAlias.h"

namespace {

struct Form
{
    IndexMetric g;
    double operator()(double a, double b, double c) const
    {
        return g.g11 * a * a + g.g22 * b * b + g.g33 * c * c + 2.0 * (g.g12 * a * b + g.g13 * a * c + g.g23 * b * c);
    }
};

int centered(int i, int n) { return (i > n / 2) ? i - n : i; }

// Reciprocal metric (index space) of a cell with lengths L and angles in radians
IndexMetric metric_of(double a, double b, double c, double alpha, double beta, double gamma)
{
    const double ca = std::cos(alpha), cb = std::cos(beta), cg = std::cos(gamma), sg = std::sin(gamma);
    const double vol = a * b * c * std::sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg);
    const double A[3][3] = {{a, 0.0, 0.0}, {b * cg, b * sg, 0.0},
                            {c * cb, c * (ca - cb * cg) / sg, vol / (a * b * sg)}};
    double R[3][3];   // rows: reciprocal vectors (inverse transpose of A)
    for (int i = 0; i < 3; ++i)
    {
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        R[i][0] = (A[j][1] * A[k][2] - A[j][2] * A[k][1]) / vol;
        R[i][1] = (A[j][2] * A[k][0] - A[j][0] * A[k][2]) / vol;
        R[i][2] = (A[j][0] * A[k][1] - A[j][1] * A[k][0]) / vol;
    }
    auto dot = [&](int i, int j) { return R[i][0] * R[j][0] + R[i][1] * R[j][1] + R[i][2] * R[j][2]; };
    return {dot(0, 0), dot(1, 1), dot(2, 2), dot(0, 1), dot(0, 2), dot(1, 2)};
}

// Former window: one period on every axis with N > 1
std::array<int, 3> one_period_rep(std::array<int, 3> m, const std::array<int, 3>& n, const Form& form)
{
    const int s[3] = {n[0] > 1, n[1] > 1, n[2] > 1};
    std::array<int, 3> best_rep = m;
    double best = form(m[0], m[1], m[2]);
    for (int a = -s[0]; a <= s[0]; ++a)
        for (int b = -s[1]; b <= s[1]; ++b)
            for (int c = -s[2]; c <= s[2]; ++c)
            {
                if (a == 0 && b == 0 && c == 0)
                    continue;
                const std::array<int, 3> t = {m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]};
                const double f = form(t[0], t[1], t[2]);
                if (f < best)
                {
                    best = f;
                    best_rep = t;
                }
            }
    return best_rep;
}

struct Stats
{
    long modes = 0, wrong_min = 0, odd = 0, wrong_ties = 0, window_differs = 0, window_misses = 0;
};

void check_grid(const std::array<int, 3>& n, const IndexMetric& g, bool expect_window, Stats& s)
{
    const Form form{g};
    const int W = 16;   // brute-force window in periods
    for (int i = 0; i < n[0]; ++i)
        for (int j = 0; j < n[1]; ++j)
            for (int k = 0; k < n[2]; ++k)
            {
                const std::array<int, 3> m = {centered(i, n[0]), centered(j, n[1]), centered(k, n[2])};
                double brute = form(m[0], m[1], m[2]);
                const int w[3] = {n[0] > 1 ? W : 0, n[1] > 1 ? W : 0, n[2] > 1 ? W : 0};
                for (int a = -w[0]; a <= w[0]; ++a)
                    for (int b = -w[1]; b <= w[1]; ++b)
                        for (int c = -w[2]; c <= w[2]; ++c)
                            brute = std::min(brute, form(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]));
                int brute_ties = 0;
                for (int a = -w[0]; a <= w[0]; ++a)
                    for (int b = -w[1]; b <= w[1]; ++b)
                        for (int c = -w[2]; c <= w[2]; ++c)
                            if (form(m[0] + a * n[0], m[1] + b * n[1], m[2] + c * n[2]) <= brute + 1e-10 * brute)
                                ++brute_ties;

                std::array<int, 3> rep = m;
                min_form_alias_rep(rep, n, g, form);
                const double f = form(rep[0], rep[1], rep[2]);
                std::array<int, 3> rep_neg = {centered((n[0] - i) % n[0], n[0]), centered((n[1] - j) % n[1], n[1]),
                                              centered((n[2] - k) % n[2], n[2])};
                min_form_alias_rep(rep_neg, n, g, form);
                const double f_neg = form(rep_neg[0], rep_neg[1], rep_neg[2]);
                const int ties = for_each_min_form_alias(m, n, g, form, 1e-10, [](int, int, int) {});
                const std::array<int, 3> window = one_period_rep(m, n, form);

                ++s.modes;
                s.wrong_min += (std::abs(f - brute) > 1e-12 * std::max(brute, 1e-300));
                s.odd += (std::abs(f - f_neg) > 1e-12 * std::max(f, 1e-300));
                s.wrong_ties += (ties != brute_ties);
                if (expect_window)
                    s.window_differs += (window != rep);
                s.window_misses += (form(window[0], window[1], window[2]) > brute * (1.0 + 1e-12));
            }
}

}  // namespace

int main()
{
    const double deg = 3.14159265358979323846 / 180.0;
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    bool ok = true;

    // 1) Random oblique in-plane cells (alpha = beta = 90) and triclinic cells,
    //    with grid spacings up to a factor 8 apart
    Stats random_cells;
    for (int trial = 0; trial < 60; ++trial)
    {
        const bool triclinic = (trial % 3 == 2);
        const std::array<int, 3> n = {2 + static_cast<int>(uni(rng) * 30), 2 + static_cast<int>(uni(rng) * 30),
                                      triclinic ? 2 + static_cast<int>(uni(rng) * 10) : 1};
        const double gamma = (50.0 + 80.0 * uni(rng)) * deg;
        const double alpha = triclinic ? (70.0 + 40.0 * uni(rng)) * deg : 90.0 * deg;
        const double beta = triclinic ? (70.0 + 40.0 * uni(rng)) * deg : 90.0 * deg;
        IndexMetric g = metric_of(1.0 + 7.0 * uni(rng), 1.0 + 7.0 * uni(rng), 1.0 + 3.0 * uni(rng), alpha, beta, gamma);
        if (!triclinic)
            g.g33 = g.g13 = g.g23 = 0.0;   // in-plane search, as in ObliqueZ
        check_grid(n, g, false, random_cells);
    }
    std::printf("random cells: %ld modes, least form wrong %ld, f(m) != f(-m) %ld, tie count wrong %ld "
                "(one-period window misses the minimum on %ld)\n",
                random_cells.modes, random_cells.wrong_min, random_cells.odd, random_cells.wrong_ties,
                random_cells.window_misses);
    ok &= (random_cells.wrong_min == 0 && random_cells.odd == 0 && random_cells.wrong_ties == 0);

    // 2) Cells with nearly isotropic spacing: same representative as the former window
    Stats isotropic;
    const std::array<double, 4> gammas = {60.0, 90.0 + 1e-6, 100.0, 120.0};
    for (double gam : gammas)
    {
        check_grid({24, 24, 1}, metric_of(2.0, 2.0, 1.0, 90 * deg, 90 * deg, gam * deg), true, isotropic);
        check_grid({12, 10, 1}, metric_of(1.5, 1.3, 1.0, 90 * deg, 90 * deg, gam * deg), true, isotropic);
        check_grid({16, 16, 12}, metric_of(2.0, 2.0, 1.6, 80 * deg, 95 * deg, gam * deg), true, isotropic);
    }
    std::printf("isotropic spacing: %ld modes, representative differs from the one-period window %ld, "
                "least form wrong %ld\n", isotropic.modes, isotropic.window_differs, isotropic.wrong_min);
    ok &= (isotropic.window_differs == 0 && isotropic.wrong_min == 0);

    // 3) 4x32 points on a 120-degree cell with equal lengths: the window fails, the search does not
    Stats aniso;
    IndexMetric g_hex = metric_of(4.0, 4.0, 2.0, 90 * deg, 90 * deg, 120 * deg);
    g_hex.g33 = g_hex.g13 = g_hex.g23 = 0.0;
    check_grid({4, 32, 1}, g_hex, false, aniso);
    std::printf("4x32 on a 4x4 120-degree cell: window misses %ld of %ld modes; search: least form wrong %ld, "
                "f(m) != f(-m) %ld\n", aniso.window_misses, aniso.modes, aniso.wrong_min, aniso.odd);
    ok &= (aniso.window_misses > 0 && aniso.wrong_min == 0 && aniso.odd == 0);

    std::printf(ok ? "\nAll passed!\n" : "\nFAILED!\n");
    return ok ? 0 : 1;
}
