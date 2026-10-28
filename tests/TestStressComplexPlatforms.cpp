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

// Stress of complex fields on every available platform (1D, 2D and a 64^3 3D
// grid) for RQM4, RK2 and the discrete chain. The platforms must agree to
// round-off, and RK2 must agree with RQM4 to its O(ds^2) splitting error. The
// fields are smooth (a few low-frequency cosines), so the methods are in their
// asymptotic regime; the 3D grid is large enough that an in-place k -> -k
// gather on the GPU (several thread blocks per k-point pair) corrupts the result.

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <complex>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "TestCudaPart.h"
#include "TestNanSafe.h"

typedef std::complex<double> T;

struct GridCase
{
    std::string name;
    std::vector<int> nx;
    std::vector<double> lx;
};

static std::vector<T> run_case(const std::string& platform, const GridCase& grid,
                               const std::string& chain_model, const std::string& numerical_method,
                               const std::vector<T>& w, double ds, T& partition)
{
    std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.2}};
    std::vector<BlockInput> blocks = {{"A", 0.4, 0, 1}, {"B", 0.6, 1, 2}};
    const int M = w.size()/2;

    std::unique_ptr<AbstractFactory<T>> factory = PlatformSelector::create_factory_complex(platform, false);
    std::unique_ptr<ComputationBox<T>> cb = factory->create_computation_box(grid.nx, grid.lx, {});
    std::unique_ptr<Molecules> molecules = factory->create_molecules_information(chain_model, ds, bond_lengths);
    molecules->add_polymer(1.0, blocks, {});
    auto optimizer = std::make_unique<PropagatorComputationOptimizer>(molecules.get(), true);
    std::unique_ptr<PropagatorComputation<T>> solver = factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), numerical_method);

    solver->compute_propagators({{"A", &w[0]}, {"B", &w[M]}}, {});
    partition = solver->get_total_partition(0);
    solver->compute_stress();
    std::vector<T> stress = solver->get_stress();

    return stress;
}

// Smooth periodic field: three cosines per axis with random amplitudes and
// phases (real part amplitude ~1, imaginary part ~0.1)
static std::vector<T> smooth_field(const std::vector<int>& nx, unsigned seed)
{
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> amp(-1.0, 1.0);
    std::uniform_real_distribution<double> phase(0.0, 2.0*M_PI);
    const int dim = nx.size();
    const int n_modes = 3;
    std::vector<double> a(dim*n_modes), p(dim*n_modes), b(dim*n_modes), q(dim*n_modes);
    for (size_t i = 0; i < a.size(); ++i)
    {
        a[i] = amp(engine); p[i] = phase(engine);
        b[i] = 0.1*amp(engine); q[i] = phase(engine);
    }
    int M = 1;
    for (int n : nx) M *= n;
    std::vector<T> w(M);
    for (int i = 0; i < M; ++i)
    {
        // grid index of each axis (last axis fastest)
        std::vector<int> idx(dim);
        int rest = i;
        for (int d = dim-1; d >= 0; --d) { idx[d] = rest % nx[d]; rest /= nx[d]; }
        double re = 0.0, im = 0.0;
        for (int d = 0; d < dim; ++d)
            for (int k = 1; k <= n_modes; ++k)
            {
                const double x = 2.0*M_PI*k*idx[d]/nx[d];
                re += a[d*n_modes + k-1]*std::cos(x + p[d*n_modes + k-1]);
                im += b[d*n_modes + k-1]*std::cos(x + q[d*n_modes + k-1]);
            }
        w[i] = T(re, im);
    }
    return w;
}

// Relative difference of the first n entries (the length derivatives)
static double rel_diff(const std::vector<T>& a, const std::vector<T>& b, int n)
{
    double scale = 0.0, diff = 0.0;
    for (int i = 0; i < n; ++i)
    {
        // std::max drops a NaN; return it so that no tolerance accepts it
        const double d = std::abs(a[i] - b[i]);
        if (!std::isfinite(d))
            return d;
        scale = max_or_nan(scale, std::abs(a[i]));
        diff = max_or_nan(diff, d);
    }
    return diff / std::max(scale, 1e-12);
}

int main()
{
    try
    {
        const double ds = 1.0/20.0;
        std::vector<GridCase> grids = {
            {"1D", {64},         {3.2}},
            {"2D", {32, 28},     {3.2, 3.4}},
            {"3D", {64, 64, 64}, {3.1, 3.3, 3.5}},
        };
        std::vector<std::pair<std::string, std::string>> models = {
            {"Continuous", "rqm4"}, {"Continuous", "rk2"}, {"Discrete", "rqm4"},
        };
        std::vector<std::string> platforms = PlatformSelector::avail_platforms();
        std::cout << "Platforms:";
        for (const auto& p : platforms) std::cout << " " << p;
        std::cout << std::endl;

        bool ok = true;
        for (const GridCase& grid : grids)
        {
            const int dim = grid.nx.size();
            std::vector<T> w = smooth_field(grid.nx, 42);
            std::vector<T> w_b = smooth_field(grid.nx, 43);
            w.insert(w.end(), w_b.begin(), w_b.end());

            std::map<std::string, std::vector<T>> stress_of_method;   // on the first platform
            for (const auto& model : models)
            {
                std::vector<T> reference;
                T partition_reference{};
                for (const std::string& platform : platforms)
                {
                    T partition;
                    std::vector<T> stress = run_case(platform, grid, model.first, model.second, w, ds, partition);
                    std::cout << grid.name << " " << model.first << " " << model.second << " " << platform
                              << ": Q = " << std::setprecision(12) << partition << ", stress =";
                    for (int d = 0; d < (int)grid.nx.size(); ++d) std::cout << " " << stress[d];
                    std::cout << std::endl;

                    if (reference.empty())
                    {
                        reference = stress;
                        partition_reference = partition;
                        continue;
                    }
                    double rel_s = rel_diff(reference, stress, dim);
                    double rel_q = std::abs(partition - partition_reference) / std::abs(partition_reference);
                    std::cout << "    vs " << platforms[0] << ": stress rel diff " << rel_s
                              << ", Q rel diff " << rel_q << std::endl;
                    if (!(rel_s <= 1e-8) || !(rel_q <= 1e-10))
                    {
                        std::cout << "ERROR: platforms disagree" << std::endl;
                        ok = false;
                    }
                }
                stress_of_method[model.first + "/" + model.second] = reference;
            }

            // RK2 and RQM4 differ by the O(ds^2) splitting error only
            double rel_m = rel_diff(stress_of_method["Continuous/rqm4"], stress_of_method["Continuous/rk2"], dim);
            std::cout << grid.name << ": RK2 vs RQM4 stress rel diff " << rel_m << std::endl;
            if (!(rel_m <= 5e-2))
            {
                std::cout << "ERROR: RK2 and RQM4 stresses disagree" << std::endl;
                ok = false;
            }
        }

        if (!ok)
        {
            std::cout << "Test failed!" << std::endl;
            return -1;
        }
        std::cout << "Test passed!" << std::endl;
        return cuda_part_missing() ? 1 : 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
