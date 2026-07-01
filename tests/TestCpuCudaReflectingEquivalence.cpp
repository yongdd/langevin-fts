/**
 * @file TestCpuCudaReflectingEquivalence.cpp
 * @brief CPU vs CUDA equivalence for reflecting (DCT) boundary conditions.
 *
 * The pseudo-spectral propagator with reflecting BC (DCT) must give identical
 * results on CPU and CUDA for BOTH the continuous and discrete chain models, in
 * every dimension. This directly exercises the C++ solvers (no Python layer), so
 * it cannot silently no-op.
 *
 * Regression guard for the CUDA discrete-chain non-periodic bug: the discrete
 * solver used to always take the periodic cuFFT path, returning wrong numbers for
 * reflecting BC (partition mismatch of 6-40% vs CPU across 1D/2D/3D).
 *
 * Skips gracefully when either cpu-fftw or cuda is not available in the build.
 */
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <random>
#include <algorithm>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

static double compute_Q(const std::string& platform, const std::string& chain_model,
                        const std::vector<int>& nx, const std::vector<double>& lx,
                        double ds, const std::vector<double>& w)
{
    AbstractFactory<double>* factory = PlatformSelector::create_factory_real(platform, false);

    std::map<std::string, double> bond_lengths = {{"A", 1.0}};
    Molecules* molecules = factory->create_molecules_information(chain_model, ds, bond_lengths);

    std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};
    molecules->add_polymer(1.0, blocks);

    PropagatorComputationOptimizer* optimizer =
        new PropagatorComputationOptimizer(molecules, false);

    std::vector<std::string> bc(2 * nx.size(), "reflecting");
    ComputationBox<double>* cb = factory->create_computation_box(nx, lx, bc);

    PropagatorComputation<double>* solver =
        factory->create_propagator_computation(cb, molecules, optimizer, "rqm4");
    solver->compute_propagators({{"A", w.data()}});
    double Q = solver->get_total_partition(0);

    delete solver;
    delete cb;
    delete optimizer;
    delete molecules;
    delete factory;
    return Q;
}

int main()
{
    try
    {
        std::vector<std::string> platforms = PlatformSelector::avail_platforms();
        bool has_cpu  = std::find(platforms.begin(), platforms.end(), "cpu-fftw") != platforms.end();
        bool has_cuda = std::find(platforms.begin(), platforms.end(), "cuda")     != platforms.end();
        if (!has_cpu || !has_cuda)
        {
            std::cout << "cpu-fftw and/or cuda not available; skipping." << std::endl;
            return 0;
        }

        struct Case { std::string chain; std::vector<int> nx; std::vector<double> lx; };
        std::vector<Case> cases = {
            {"Continuous", {32},          {3.0}},
            {"Discrete",   {32},          {3.0}},
            {"Continuous", {16, 16},      {3.0, 3.0}},
            {"Discrete",   {16, 16},      {3.0, 3.0}},
            {"Continuous", {16, 16, 16},  {3.0, 3.0, 3.0}},
            {"Discrete",   {16, 16, 16},  {3.0, 3.0, 3.0}},
        };
        const double ds = 1.0 / 50.0;
        const double rtol = 1e-9;

        bool all_ok = true;
        std::cout << std::fixed << std::setprecision(12);
        for (const auto& c : cases)
        {
            int M = 1;
            for (int n : c.nx) M *= n;

            // Deterministic random field, std ~ 2 (same for both platforms).
            std::vector<double> w(M);
            std::mt19937 gen(12345);
            std::normal_distribution<double> dist(0.0, 2.0);
            for (int i = 0; i < M; ++i) w[i] = dist(gen);

            double Qcpu  = compute_Q("cpu-fftw", c.chain, c.nx, c.lx, ds, w);
            double Qcuda = compute_Q("cuda",     c.chain, c.nx, c.lx, ds, w);
            double rel = std::abs(Qcpu - Qcuda) / std::abs(Qcpu);
            bool pass = rel < rtol;
            all_ok = all_ok && pass;

            std::cout << (pass ? "  PASS  " : "  FAIL  ")
                      << c.chain << " " << c.nx.size() << "D reflecting"
                      << "  Qcpu=" << Qcpu << "  Qcuda=" << Qcuda
                      << "  rel=" << std::scientific << rel << std::fixed << std::endl;
        }

        if (all_ok)
        {
            std::cout << "All CPU/CUDA reflecting-BC equivalence tests PASSED." << std::endl;
            return 0;
        }
        std::cout << "CPU/CUDA reflecting-BC equivalence FAILURES detected." << std::endl;
        return -1;
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
}
