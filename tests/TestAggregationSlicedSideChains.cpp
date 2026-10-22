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

// Aggregation of side chains whose substituted right keys are merged again at a
// later left key, closer to the chain ends. When identical side chains are
// aggregated, the right keys of the blocks next to them are rewritten
// (substituted). If a later left key merges such a substituted key with others,
// only the merged key may rewrite the blocks beyond it, not the sliced key
// itself. Each run must reproduce the concentrations and partition function of
// the run without aggregation, on every platform, with and without
// reduce_memory, and aggregation must still reduce the number of computed
// segments.

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "TestNanSafe.h"

struct RunResult
{
    std::vector<double> phi;    // phi_A followed by phi_B
    double partition;
    double mass_error;
    bool partition_check;
    int n_segments;             // total segments of the propagators to compute
};

static RunResult run(const std::string& platform, bool reduce_memory, bool aggregate,
                     const std::string& chain_model, double ds, const std::vector<BlockInput>& blocks,
                     const std::vector<int>& nx, const std::vector<double>& lx, const std::vector<double>& w)
{
    const int M = w.size()/2;
    std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.2}};

    std::unique_ptr<AbstractFactory<double>> factory(PlatformSelector::create_factory_real(platform, reduce_memory));
    std::unique_ptr<ComputationBox<double>> cb(factory->create_computation_box(nx, lx, {}));
    std::unique_ptr<Molecules> molecules(factory->create_molecules_information(chain_model, ds, bond_lengths));
    molecules->add_polymer(1.0, blocks, {});
    std::unique_ptr<PropagatorComputationOptimizer> optimizer(new PropagatorComputationOptimizer(molecules.get(), aggregate));
    std::unique_ptr<PropagatorComputation<double>> solver(
        factory->create_propagator_computation(cb.get(), molecules.get(), optimizer.get(), "rqm4"));

    RunResult result;
    result.n_segments = 0;
    for(const auto& item : optimizer->get_computation_propagators())
        result.n_segments += item.second.max_n_segment;

    solver->compute_propagators({{"A", &w[0]}, {"B", &w[M]}}, {});
    solver->compute_concentrations();
    result.phi.resize(2*M);
    solver->get_total_concentration("A", &result.phi[0]);
    solver->get_total_concentration("B", &result.phi[M]);
    result.partition = solver->get_total_partition(0);
    result.partition_check = solver->check_total_partition();

    std::vector<double> phi_total(M);
    for(int i=0; i<M; i++)
        phi_total[i] = result.phi[i] + result.phi[i+M];
    result.mass_error = cb->integral(phi_total.data())/cb->get_volume() - 1.0;
    return result;
}

int main()
{
    try
    {
        const double ds = 0.02;
        const std::vector<int> nx = {12, 10};
        const std::vector<double> lx = {1.5, 1.3};
        const int M = nx[0]*nx[1];

        // A backbone of 1, 2, 3 and 1 segments with a B2-A1 side chain on every
        // backbone node (without the fix: wrong concentrations of the discrete
        // chain, an exception of the continuous chain)
        std::vector<BlockInput> brush;
        {
            const std::vector<int> backbone = {1, 2, 3, 1};
            for(size_t i=0; i<backbone.size(); i++)
                brush.push_back({"A", ds*backbone[i], (int) i, (int) i+1});
            int node = 100;
            for(size_t i=0; i<=backbone.size(); i++)
            {
                brush.push_back({"B", ds*2, (int) i, node});
                brush.push_back({"A", ds*1, node, node+1});
                node += 2;
            }
        }
        // Three arms B3-B3-A1, B2-B3-A1 and B2-A1 on node 0 (without the fix:
        // wrong concentrations of the discrete chain)
        std::vector<BlockInput> branched =
        {
            {"B", ds*3, 0, 1}, {"B", ds*3, 1, 2}, {"A", ds*1, 2, 3},
            {"B", ds*2, 0, 4}, {"B", ds*3, 4, 5}, {"A", ds*1, 5, 6},
            {"B", ds*2, 0, 7}, {"A", ds*1, 7, 8},
        };
        std::vector<std::pair<std::string, std::vector<BlockInput>>> polymers =
            {{"brush", brush}, {"branched", branched}};

        // Random fields with standard deviation 5
        std::mt19937 generator(1234);
        std::normal_distribution<double> normal(0.0, 5.0);
        std::vector<double> w(2*M);
        for(double& value : w)
            value = normal(generator);

        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        bool passed = true;
        for(const auto& [name, blocks] : polymers)
        {
            for(std::string chain_model : {"Continuous", "Discrete"})
            {
                // Reference: first platform, full storage, no aggregation
                RunResult ref = run(avail_platforms[0], false, false, chain_model, ds, blocks, nx, lx, w);
                for(const std::string& platform : avail_platforms)
                {
                    for(bool reduce_memory : {false, true})
                    {
                        for(bool aggregate : {false, true})
                        {
                            RunResult r;
                            try
                            {
                                r = run(platform, reduce_memory, aggregate, chain_model, ds, blocks, nx, lx, w);
                            }
                            catch(std::exception& exc)
                            {
                                std::cout << std::setw(9) << name << " " << std::setw(10) << chain_model
                                          << " " << std::setw(9) << platform
                                          << " reduce_memory=" << reduce_memory << " aggregate=" << aggregate
                                          << "  FAILED: " << exc.what() << std::endl;
                                passed = false;
                                continue;
                            }
                            std::vector<double> diff(2*M);
                            for(int i=0; i<2*M; i++)
                                diff[i] = r.phi[i] - ref.phi[i];
                            double phi_error = max_abs_or_nan(diff);
                            double partition_error = std::abs(r.partition - ref.partition)/std::abs(ref.partition);
                            bool fewer_segments = !aggregate || r.n_segments < ref.n_segments;
                            bool ok = r.partition_check && fewer_segments
                                && phi_error <= 1e-12 && partition_error <= 1e-12 && std::abs(r.mass_error) <= 1e-12;

                            std::cout << std::setw(9) << name << " " << std::setw(10) << chain_model
                                      << " " << std::setw(9) << platform
                                      << " reduce_memory=" << reduce_memory << " aggregate=" << aggregate
                                      << std::scientific << std::setprecision(2)
                                      << "  max|dphi|=" << phi_error << "  |dQ|/Q=" << partition_error
                                      << "  mass error=" << r.mass_error
                                      << "  segments=" << r.n_segments << "/" << ref.n_segments
                                      << "  check=" << r.partition_check
                                      << (ok ? "" : "  FAILED") << std::endl;
                            passed = passed && ok;
                        }
                    }
                }
            }
        }
        if(!passed)
            return -1;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
