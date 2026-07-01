/**
 * @file TestCpuCudaNonPeriodicEquivalence.cpp
 * @brief Cross-backend equivalence for non-periodic (DCT/DST) boundary conditions.
 *
 * The pseudo-spectral propagator with reflecting (DCT) and absorbing (DST) BC must
 * give identical partition functions across all available real backends
 * (cpu-mkl, cpu-fftw, cuda) for both the continuous and discrete chain models, in
 * 1D/2D/3D. Random fields (std ~ 2) contain Nyquist-frequency content, which is what
 * exposed the transform bugs below.
 *
 * Regression guard for:
 *   - CUDA discrete non-periodic path that used the periodic cuFFT plans (wrong DCT/DST).
 *   - CudaFFT non-periodic glue kernels missing grid-stride loops (NaN for M > 65536,
 *     hence the 48^3 case).
 *   - MklFFT DST-III backward dropping the highest (Nyquist) mode (absorbing BC).
 *
 * Skips gracefully when fewer than two real backends are available.
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
                        const std::string& bc_kind, double ds,
                        const std::vector<double>& w)
{
    AbstractFactory<double>* factory = PlatformSelector::create_factory_real(platform, false);

    std::map<std::string, double> bond_lengths = {{"A", 1.0}};
    Molecules* molecules = factory->create_molecules_information(chain_model, ds, bond_lengths);

    std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};
    molecules->add_polymer(1.0, blocks);

    PropagatorComputationOptimizer* optimizer =
        new PropagatorComputationOptimizer(molecules, false);

    std::vector<std::string> bc(2 * nx.size(), bc_kind);
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
        std::vector<std::string> avail = PlatformSelector::avail_platforms();
        std::vector<std::string> platforms;
        for (const std::string& p : {std::string("cpu-mkl"), std::string("cpu-fftw"), std::string("cuda")})
            if (std::find(avail.begin(), avail.end(), p) != avail.end())
                platforms.push_back(p);
        if (platforms.size() < 2)
        {
            std::cout << "Fewer than two real backends available; skipping." << std::endl;
            return 0;
        }

        struct Case { std::string chain; std::string bc; std::vector<int> nx; std::vector<double> lx; };
        std::vector<Case> cases;
        for (const std::string& chain : {std::string("Continuous"), std::string("Discrete")})
            for (const std::string& bc : {std::string("reflecting"), std::string("absorbing")})
            {
                cases.push_back({chain, bc, {32},         {3.0}});
                cases.push_back({chain, bc, {16, 16},     {3.0, 3.0}});
                cases.push_back({chain, bc, {16, 16, 16}, {3.0, 3.0, 3.0}});
                // M = 48^3 = 110592 > n_blocks*n_threads (65536): exercises the
                // CudaFFT non-periodic glue-kernel grid-stride loops.
                cases.push_back({chain, bc, {48, 48, 48}, {3.0, 3.0, 3.0}});
            }
        const double ds = 1.0 / 50.0;
        const double rtol = 1e-9;

        bool all_ok = true;
        std::cout << std::fixed << std::setprecision(12);
        for (const auto& c : cases)
        {
            int M = 1;
            for (int n : c.nx) M *= n;

            std::vector<double> w(M);
            std::mt19937 gen(12345);
            std::normal_distribution<double> dist(0.0, 2.0);
            for (int i = 0; i < M; ++i) w[i] = dist(gen);

            double Qref = compute_Q(platforms[0], c.chain, c.nx, c.lx, c.bc, ds, w);
            double maxrel = 0.0;
            std::string worst;
            for (size_t p = 1; p < platforms.size(); ++p)
            {
                double Q = compute_Q(platforms[p], c.chain, c.nx, c.lx, c.bc, ds, w);
                double rel = std::isfinite(Q) ? std::abs(Q - Qref) / std::abs(Qref) : 1e300;
                if (rel > maxrel) { maxrel = rel; worst = platforms[p]; }
            }
            bool pass = maxrel < rtol;
            all_ok = all_ok && pass;

            std::cout << (pass ? "  PASS  " : "  FAIL  ")
                      << c.chain << " " << c.nx.size() << "D " << c.bc
                      << "  Qref(" << platforms[0] << ")=" << Qref
                      << "  max_rel=" << std::scientific << maxrel << std::fixed
                      << (worst.empty() ? "" : "  (vs " + worst + ")") << std::endl;
        }

        if (all_ok)
        {
            std::cout << "All non-periodic CPU/CUDA equivalence tests PASSED." << std::endl;
            return 0;
        }
        std::cout << "Non-periodic CPU/CUDA equivalence FAILURES detected." << std::endl;
        return -1;
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
}
