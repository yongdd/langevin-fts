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

#include <iostream>
#include <cmath>
#include <numbers>
#include <map>
#include <string>
#include <vector>

#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "Scheduler.h"

int main()
{
    try
    {
        // std::vector<std::string> block_monomer_types = {"A","A","B","B","A","A","B","A","B","B","A","A","B","A","B","A","A","B","A"};
        // std::vector<double> contour_lengths = {0.4,1.2,1.2,0.9,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,0.9,1.2,1.2,1.2,0.9,0.9};
        // std::vector<int> v = {0,0,0,0,1,1,2,2,2,3,4,4,7,8,9,9,10,13,13};
        // std::vector<int> u = {1,2,5,6,4,15,3,7,10,14,8,9,19,13,12,16,11,17,18};

        // for(size_t p=0; p<block_monomer_types.size(); p++){
        //     printf("{\"%s\",%4.1f,%2d,%2d},\n", block_monomer_types[p].c_str(), contour_lengths[p], v[p], u[p]);
        // }

        std::vector<BlockInput> blocks =
        {
            {"A", 0.4, 0, 1},
            {"A", 1.2, 0, 2},
            {"B", 1.2, 0, 5},
            {"B", 0.9, 0, 6},
            {"A", 0.9, 1, 4},
            {"A", 1.2, 1,15},
            {"B", 1.2, 2, 3},
            {"A", 0.9, 2, 7},
            {"B", 1.2, 2,10},
            {"B", 1.2, 3,14},
            {"A", 0.9, 4, 8},
            {"A", 1.2, 4, 9},
            {"B", 1.2, 7,19},
            {"A", 0.9, 8,13},
            {"B", 1.2, 9,12},
            {"A", 1.2, 9,16},
            {"A", 1.2,10,11},
            {"B", 0.9,13,17},
            {"A", 0.9,13,18},
        };

        Molecules molecules("Continuous", 0.1, {{"A",1.0}, {"B",2.0}});
        molecules.add_polymer(1.0, blocks, {});

        PropagatorComputationOptimizer propagator_computation_optimizer(&molecules, false);

        Scheduler sc(propagator_computation_optimizer.get_computation_propagators(), 4);

        sc.display(propagator_computation_optimizer.get_computation_propagators());

        // The entries of the schedule run one after the other and the jobs of an entry in parallel.
        // Every propagator must be advanced through all its segments once, in order, at most once per
        // entry, and start only after an earlier entry has computed the segments of its dependencies
        // that its initial condition needs; an entry holds at most one job per stream
        auto propagators = propagator_computation_optimizer.get_computation_propagators();
        bool ok = true;
        auto fail = [&ok](const std::string& message) { std::cout << "FAILED: " << message << std::endl; ok = false; };
        for (int n_streams = 1; n_streams <= 4; ++n_streams)
        {
            Scheduler scheduler(propagators, n_streams);
            const auto& schedule = scheduler.get_schedule();
            std::map<std::string, int> first_entry, last_entry, reached;
            std::map<std::string, std::vector<std::pair<int, int>>> progress;   // (entry, segments computed)
            for (int e = 0; e < static_cast<int>(schedule.size()); ++e)
            {
                if (static_cast<int>(schedule[e].size()) > n_streams)
                    fail("entry " + std::to_string(e) + " holds more jobs than streams");
                for (const auto& [key, from, to] : schedule[e])
                {
                    if (first_entry.find(key) == first_entry.end())
                    {
                        if (from != 0)
                            fail(key + " starts at segment " + std::to_string(from));
                        first_entry[key] = e;
                    }
                    else if (last_entry[key] >= e || from != reached[key])
                        fail(key + " is not advanced in order in entry " + std::to_string(e));
                    last_entry[key] = e;
                    reached[key] = to;
                    progress[key].emplace_back(e, to);
                }
            }
            for (const auto& [key, edge] : propagators)
            {
                if (first_entry.find(key) == first_entry.end() || reached[key] != edge.max_n_segment)
                {
                    fail(key + " is not computed through its " + std::to_string(edge.max_n_segment) + " segments");
                    continue;
                }
                for (const auto& [dep_key, dep_n_segment, dep_n_repeated] : edge.deps)
                {
                    int done = -1;
                    for (const auto& [e, to] : progress[dep_key])
                        if (to >= dep_n_segment) { done = e; break; }
                    if (done < 0 || done >= first_entry[key])
                        fail(key + " starts before segment " + std::to_string(dep_n_segment) + " of " + dep_key
                             + " is computed (" + std::to_string(n_streams) + " streams)");
                }
            }
        }
        if (!ok)
            return 1;
        std::cout << "Schedules with 1 to 4 streams respect the dependencies" << std::endl;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
