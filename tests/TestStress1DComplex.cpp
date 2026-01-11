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
        std::vector<int> nx = {64};
        std::vector<double> lx = {3.3};
        double ds = 1.0/100.0;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};

        // Simple diblock - same structure for both polymers
        std::vector<BlockInput> blocks_1 =
        {
            {"A", 0.5, 0, 1},
            {"B", 0.5, 1, 2},
        };
        std::vector<BlockInput> blocks_2 = blocks_1;

        const int M = nx[0];
        w = new T[2*M];

        // Use random complex fields - consistency check doesn't require SCFT
        std::mt19937 engine(static_cast<unsigned int>(42));
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < 2*M; ++i) {
            w[i] = T(dist(engine), dist(engine) * 0.1);  // Small imaginary part
        }

        std::vector<T> stress_list{};

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        std::vector<bool> aggregate_propagator_computations = {false, true};

        for(std::string platform : avail_platforms)
        {
            if(platform != "cpu-mkl")
                continue;

            for(std::string chain_model : chain_models)
            {
                std::cout << "Testing: " << chain_model << std::endl;
                std::vector<T> model_stress_list{};

                for(bool aggregate_propagator_computation : aggregate_propagator_computations)
                {
                    bool reduce_memory_usage = false;
                    AbstractFactory<T> *factory = PlatformSelector::create_factory_complex(platform, reduce_memory_usage);

                    // Create instances
                    ComputationBox<T>* cb = factory->create_computation_box(nx, lx, {});
                    Molecules* molecules = factory->create_molecules_information(chain_model, ds, bond_lengths);
                    molecules->add_polymer(0.5, blocks_1, {});
                    molecules->add_polymer(0.5, blocks_2, {});
                    PropagatorComputationOptimizer* propagator_computation_optimizer =
                        new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
                    PropagatorComputation<T>* solver =
                        factory->create_pseudospectral_solver(cb, molecules, propagator_computation_optimizer, "rqm4");

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
                    std::cout << "  Stress: " << stress[0] << std::endl;
                    model_stress_list.push_back(stress[0]);
                    stress_list.push_back(stress[0]);

                    delete molecules;
                    delete propagator_computation_optimizer;
                    delete cb;
                    delete solver;
                    delete factory;
                }

                // Check consistency within same chain model
                if (model_stress_list.size() >= 2) {
                    double diff = std::abs(model_stress_list[0] - model_stress_list[1]);
                    double max_val = std::max(std::abs(model_stress_list[0]), std::abs(model_stress_list[1]));
                    double rel_diff = diff / (max_val + 1e-10);
                    std::cout << "  Relative difference (aggregation): " << rel_diff << std::endl;
                    if (rel_diff > 1e-7) {
                        std::cout << "ERROR: Stress values differ too much!" << std::endl;
                        return -1;
                    }
                }
            }
        }

        delete[] w;

        // Overall consistency check
        if (stress_list.size() > 0) {
            T mean = std::accumulate(stress_list.begin(), stress_list.end(), T(0.0, 0.0)) / (double)stress_list.size();
            std::cout << "Overall mean stress: " << mean << std::endl;
        }

        std::cout << "Test passed!" << std::endl;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
