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
 * @file TestCudaCrysTablesAllocation.cu
 * @brief The CUDA CrysFFT paths allocate no tables while computing propagators.
 *
 * The Boltzmann tables of the CrysFFT steps are built when the computation is
 * created and when the cell changes (update_laplacian_operator), not lazily in
 * the propagator loop. For each CrysFFT mode (recursive 3m, Pmmm DCT, ObliqueZ)
 * and each pseudo-spectral method:
 *
 * - The first compute_propagators()/compute_concentrations() may make only the
 *   counted allocations that the same system makes with the group P1, which has
 *   no CrysFFT path (the device copies of the fields); the allocation test hook
 *   fails any further one.
 * - After a cell change, these calls may allocate nothing.
 * - Neither call builds a table (the table cache reuses the buffers of dropped
 *   tables, so only its build counter shows a build in the loop).
 * - compute_stress() builds the stress multipliers before its segment loop and
 *   only once per cell: a second call builds none and gives the same stress.
 * - After the change, the results equal those of a computation created for the
 *   new cell.
 */

#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "CudaCommon.h"
#include "CudaCrysFFTTables.h"
#include "AbstractFactory.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "PlatformSelector.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "SpaceGroup.h"

struct Case
{
    std::string name;
    std::string chain_model;
    std::string method;
    std::vector<int> nx;
    std::vector<double> lx;
    std::vector<double> angles;   ///< empty: orthogonal
    std::string sg_symbol;
    int sg_hall;
    std::string basis;            ///< "m3", "pmmm" or "z-mirror"
};

struct Computation
{
    std::unique_ptr<AbstractFactory<double>> factory;
    std::unique_ptr<SpaceGroup> sg;
    std::unique_ptr<ComputationBox<double>> cb;
    std::unique_ptr<Molecules> molecules;
    std::unique_ptr<PropagatorComputationOptimizer> optimizer;
    std::unique_ptr<PropagatorComputation<double>> computation;
    std::vector<double> w_a, w_b;
};

/// The computation of a case, or with p1 the same system with the group P1
static Computation create(const Case& c, const std::vector<double>& lx, bool p1 = false)
{
    Computation s;
    s.factory.reset(PlatformSelector::create_factory_real("cuda", false));
    s.sg = p1 ? std::make_unique<SpaceGroup>(c.nx, "P1", 1) : std::make_unique<SpaceGroup>(c.nx, c.sg_symbol, c.sg_hall);
    if (!p1 && c.basis == "m3")       s.sg->enable_m3_physical_basis();
    if (!p1 && c.basis == "pmmm")     s.sg->enable_pmmm_physical_basis();
    if (!p1 && c.basis == "z-mirror") s.sg->enable_z_mirror_physical_basis();
    const std::vector<std::string> periodic;
    s.cb.reset(c.angles.empty()
        ? s.factory->create_computation_box(c.nx, lx, periodic)
        : s.factory->create_computation_box(c.nx, lx, periodic, c.angles));
    s.molecules.reset(s.factory->create_molecules_information(c.chain_model, 0.05, {{"A", 1.0}, {"B", 1.3}}));
    s.molecules->add_polymer(1.0, {{"A", 0.4, 0, 1}, {"B", 0.6, 1, 2}});
    s.optimizer.reset(s.factory->create_propagator_computation_optimizer(s.molecules.get(), true));
    s.computation.reset(s.factory->create_propagator_computation(
        s.cb.get(), s.molecules.get(), s.optimizer.get(), c.method, s.sg.get()));

    std::mt19937 rng(43);
    std::normal_distribution<double> normal(0.0, 5.0);
    s.w_a.resize(s.cb->get_n_basis());
    s.w_b.resize(s.cb->get_n_basis());
    for (size_t i = 0; i < s.w_a.size(); ++i)
    {
        s.w_a[i] = normal(rng);
        s.w_b[i] = normal(rng);
    }
    return s;
}

static void compute(Computation& s)
{
    s.computation->compute_propagators({{"A", s.w_a.data()}, {"B", s.w_b.data()}});
    s.computation->compute_concentrations();
}

/// Number of counted allocations made by compute()
static long long count_allocations(Computation& s)
{
    const long long start = 1LL << 40;
    cuda_buffer_fail_countdown().store(start);
    compute(s);
    const long long n = start - cuda_buffer_fail_countdown().load();
    cuda_buffer_fail_countdown().store(-1);
    return n;
}

/// compute(); false if it makes more than `allowed` counted allocations
static bool compute_with_allocations(Computation& s, long long allowed)
{
    bool ok = true;
    cuda_buffer_fail_countdown().store(allowed);
    try
    {
        compute(s);
    }
    catch (std::exception& exc)
    {
        printf("  %s\n", exc.what());
        ok = false;
    }
    cuda_buffer_fail_countdown().store(-1);
    return ok;
}

/// Number of CrysFFT tables built by f()
template <typename F>
static long long count_table_builds(F&& f)
{
    const long long start = cuda_crysfft_table_build_counter().load();
    f();
    return cuda_crysfft_table_build_counter().load() - start;
}

static bool no_builds(const char* what, long long builds)
{
    if (builds != 0)
        printf("  %s built %lld CrysFFT tables\n", what, builds);
    return builds == 0;
}

int main()
{
    std::vector<Case> cases = {
        {"rqm4, Im-3m m3 basis (recursive 3m)", "continuous", "rqm4", {16, 16, 16}, {1.9, 1.9, 1.9}, {}, "Im-3m", 529, "m3"},
        {"rk2, Im-3m m3 basis (recursive 3m)", "continuous", "rk2", {16, 16, 16}, {1.9, 1.9, 1.9}, {}, "Im-3m", 529, "m3"},
        {"discrete, Pmmm basis (Pmmm DCT)", "discrete", "rqm4", {16, 16, 16}, {2.0, 1.8, 1.6}, {}, "Pmmm", -1, "pmmm"},
        {"rqm4, Pmmm basis (Pmmm DCT)", "continuous", "rqm4", {16, 16, 16}, {2.0, 1.8, 1.6}, {}, "Pmmm", -1, "pmmm"},
        {"rk2, P6_3/mmc z-mirror basis (ObliqueZ)", "continuous", "rk2", {12, 12, 16}, {1.8, 1.8, 2.4}, {90.0, 90.0, 120.0}, "P6_3/mmc", 488, "z-mirror"},
        {"discrete, P6_3/mmc z-mirror basis (ObliqueZ)", "discrete", "rqm4", {12, 12, 16}, {1.8, 1.8, 2.4}, {90.0, 90.0, 120.0}, "P6_3/mmc", 488, "z-mirror"},
    };

    bool all_ok = true;
    try
    {
        for (const Case& c : cases)
        {
            Computation p1 = create(c, c.lx, /*p1=*/true);
            const long long first_call = count_allocations(p1);

            Computation s = create(c, c.lx);
            bool ok = true;
            long long builds = count_table_builds([&] { ok = compute_with_allocations(s, first_call); });
            ok = no_builds("The first call", builds) && ok;

            // A new cell: the tables are rebuilt by update_laplacian_operator()
            std::vector<double> new_lx = c.lx;
            for (double& l : new_lx)
                l *= 1.07;
            s.cb->set_lx(new_lx);
            s.computation->update_laplacian_operator();
            builds = count_table_builds([&] { ok = compute_with_allocations(s, 0) && ok; });
            ok = no_builds("The call after the cell change", builds) && ok;
            ok = s.computation->check_total_partition() && ok;

            // Stress: multipliers before the segment loop, once per cell
            s.computation->compute_stress();
            const std::vector<double> stress = s.computation->get_stress();
            builds = count_table_builds([&] { s.computation->compute_stress(); });
            ok = no_builds("The second compute_stress()", builds) && ok;
            if (s.computation->get_stress() != stress)
            {
                printf("  The second compute_stress() changed the stress\n");
                ok = false;
            }

            Computation fresh = create(c, new_lx);
            compute(fresh);
            const double q = s.computation->get_total_partition(0);
            const double q_fresh = fresh.computation->get_total_partition(0);
            const bool same = std::abs(q - q_fresh) <= 1e-13 * std::abs(q_fresh);
            if (!same)
                printf("  Q after the cell change %.15e, fresh %.15e\n", q, q_fresh);
            ok = same && ok;

            printf("%s (%lld allocations in the first call): %s\n", c.name.c_str(), first_call, ok ? "ok" : "FAILED");
            all_ok = all_ok && ok;
        }
    }
    catch (std::exception& exc)
    {
        printf("Unexpected exception: %s\n", exc.what());
        return -1;
    }

    if (!all_ok)
    {
        printf("TestCudaCrysTablesAllocation FAILED\n");
        return -1;
    }
    printf("TestCudaCrysTablesAllocation passed\n");
    return 0;
}
