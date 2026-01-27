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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <string>
#include <array>
#include <chrono>
#include <random>
#include <numeric>
#include <complex>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

typedef std::complex<double> T;

int main()
{
    try
    {
        // Input fields
        T *w;

        // String to output file and print stream
        std::streamsize default_precision = std::cout.precision();

        // -------------- Initialize ------------
        std::vector<int> nx = {32, 28};
        std::vector<double> lx = {3.2, 3.4};
        double ds = 1.0/100.0;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};

        // Simple diblock
        std::vector<BlockInput> blocks =
        {
            {"A", 0.4, 0, 1},
            {"B", 0.6, 1, 2},
        };

        const int M = nx[0] * nx[1];
        w = new T[2*M];

        // Use random complex fields - consistency check doesn't require SCFT
        std::mt19937 engine(static_cast<unsigned int>(42));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < 2*M; ++i) {
            w[i] = T(dist(engine), dist(engine) * 0.1);  // Small imaginary part
        }

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        // Test all pseudo-spectral methods (CN-ADI methods don't support stress yet)
        std::vector<std::string> numerical_methods_discrete = {"rqm4"};
        std::vector<std::string> numerical_methods_continuous = {"rqm4", "rk2"};
        std::vector<bool> aggregate_propagator_computations = {false, true};

        for(std::string platform : avail_platforms)
        {
            if(platform != "cpu-fftw")
                continue;

            for(std::string chain_model : chain_models)
            {
                // Select numerical methods based on chain model
                const std::vector<std::string>& numerical_methods =
                    (chain_model == "Discrete") ? numerical_methods_discrete : numerical_methods_continuous;

                for(std::string numerical_method : numerical_methods)
                {
                std::cout << "Testing: " << chain_model << ", " << numerical_method << std::endl;
                std::vector<T> stress0_list{};
                std::vector<T> stress1_list{};

                for(bool aggregate_propagator_computation : aggregate_propagator_computations)
                {
                    bool reduce_memory = false;
                    AbstractFactory<T> *factory = PlatformSelector::create_factory_complex(platform, reduce_memory);

                    // Create instances
                    ComputationBox<T>* cb = factory->create_computation_box(nx, lx, {});
                    Molecules* molecules = factory->create_molecules_information(chain_model, ds, bond_lengths);
                    molecules->add_polymer(1.0, blocks, {});
                    PropagatorComputationOptimizer* propagator_computation_optimizer =
                        new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
                    PropagatorComputation<T>* solver =
                        factory->create_propagator_computation(cb, molecules, propagator_computation_optimizer, numerical_method);

                    // Display polymer architecture and propagator info (only on first aggregate iteration)
                    if (!aggregate_propagator_computation)
                    {
                        molecules->display_architectures();
                        propagator_computation_optimizer->display_blocks();
                        propagator_computation_optimizer->display_propagators();
                    }

                    std::cout << "  Aggregation: " << std::boolalpha << aggregate_propagator_computation << std::endl;

                    // Compute propagators with fixed fields
                    solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});

                    // Compute stress
                    solver->compute_stress();
                    auto stress = solver->get_stress();

                    // For 2D, stress has components: [0]=yy, [1]=zz, [2]=yz (cross-term)
                    std::cout << "  Stress[0] (yy): " << stress[0] << std::endl;
                    std::cout << "  Stress[1] (zz): " << stress[1] << std::endl;
                    stress0_list.push_back(stress[0]);
                    stress1_list.push_back(stress[1]);

                    delete molecules;
                    delete propagator_computation_optimizer;
                    delete cb;
                    delete solver;
                    delete factory;
                }

                // Check consistency within same chain model for stress[0]
                if (stress0_list.size() >= 2) {
                    double diff = std::abs(stress0_list[0] - stress0_list[1]);
                    double max_val = std::max(std::abs(stress0_list[0]), std::abs(stress0_list[1]));
                    double rel_diff = diff / (max_val + 1e-10);
                    std::cout << "  Stress[0] relative difference (aggregation): " << rel_diff << std::endl;
                    if (rel_diff > 1e-7) {
                        std::cout << "ERROR: Stress[0] values differ too much!" << std::endl;
                        return -1;
                    }
                }

                // Check consistency within same chain model for stress[1]
                if (stress1_list.size() >= 2) {
                    double diff = std::abs(stress1_list[0] - stress1_list[1]);
                    double max_val = std::max(std::abs(stress1_list[0]), std::abs(stress1_list[1]));
                    double rel_diff = diff / (max_val + 1e-10);
                    std::cout << "  Stress[1] relative difference (aggregation): " << rel_diff << std::endl;
                    if (rel_diff > 1e-7) {
                        std::cout << "ERROR: Stress[1] values differ too much!" << std::endl;
                        return -1;
                    }
                }
                }
            }
        }

        delete[] w;

        std::cout << "Test passed!" << std::endl;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
