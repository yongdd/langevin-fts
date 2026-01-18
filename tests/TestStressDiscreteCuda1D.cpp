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

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

int main()
{
    try
    {
        // Chrono timer
        std::chrono::system_clock::time_point chrono_start, chrono_end;

        // Input fields
        double *w;

        // String to output file and print stream
        std::streamsize default_precision = std::cout.precision();
        std::ofstream print_stream;
        std::stringstream ss;
        std::string print_file_name;

        // -------------- Initialize ------------
        // Platform type, [cuda, cpu-mkl]
        std::vector<int> nx = {10};
        std::vector<double> lx = {1.0};
        double ds = 1.0/10.0;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.5}};

        std::vector<BlockInput> blocks_1 =
        {
            {"A", 0.2, 0, 1},
            {"A", 0.4, 0, 2},
            {"A", 0.3, 1, 3},
            {"A", 0.4, 1, 4},

        };

        std::vector<BlockInput> blocks_2 =
        {
            {"A", 0.2, 0, 1},
            {"A", 0.4, 0, 3},
            {"A", 0.3, 1, 2},
            {"A", 0.4, 1, 4},

        };

        const int M = nx[0];
        //-------------- Allocate array ------------
        w = new double[2*M];

        // Use the random number generation to fill the array with random floats
        std::mt19937 engine(static_cast<unsigned int>(0)); 
        std::uniform_real_distribution<float> dist(-1.0, 1.0);
        for (int i = 0; i < 2*M; ++i) {
            w[i] = dist(engine);
            // w[i] = 0.0;
        }

        bool reduce_memory=false;
        std::vector<double> stress_list{};

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::string chain_model = "Discrete";
        std::vector<bool> aggregate_propagator_computations = {false, true};
        for(std::string platform : avail_platforms)
        {
            if(platform != "cuda")
                continue;

            for(bool aggregate_propagator_computation : aggregate_propagator_computations)
            {
                AbstractFactory<double> *factory = PlatformSelector::create_factory_real(platform, reduce_memory);
                factory->display_info();

                // Create instances and assign to the variables of base classes for the dynamic binding
                ComputationBox<double>* cb = factory->create_computation_box(nx, lx, {});
                Molecules* molecules        = factory->create_molecules_information(chain_model, ds, bond_lengths);
                molecules->add_polymer(0.3, blocks_1, {});
                molecules->add_polymer(0.7, blocks_2, {});
                PropagatorComputationOptimizer* propagator_computation_optimizer= new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
                PropagatorComputation<double>* solver     = factory->create_propagator_computation(cb, molecules, propagator_computation_optimizer, "rqm4");

                // -------------- Print simulation parameters ------------
                std::cout << std::setprecision(default_precision);
                std::cout << std::boolalpha;
                // std::cout<< "---------- Simulation Parameters ----------" << std::endl;
                // std::cout << "Box Dimension: " << cb->get_dim() << std::endl;
                std::cout << "Chain Model: " << molecules->get_model_name() << std::endl;
                std::cout << "Using Aggregation: " << aggregate_propagator_computation << std::endl;

                // Display polymer architecture and propagator info
                molecules->display_architectures();
                propagator_computation_optimizer->display_blocks();
                propagator_computation_optimizer->display_propagators();

                // For the given fields find the polymer statistics
                solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});

                // Compute stress
                solver->compute_stress();
                auto stress = solver->get_stress();
                std:: cout << "Stress : " << stress[0] << std::endl;
                stress_list.push_back(stress[0]);

                //------------- Finalize -------------
                delete molecules;
                delete propagator_computation_optimizer;
                delete cb;
                delete solver;
                delete factory;
            }
        }
        delete[] w;

        double mean = std::accumulate(stress_list.begin(), stress_list.end(), 0.0)/stress_list.size();
        double sq_sum = std::inner_product(stress_list.begin(), stress_list.end(), stress_list.begin(), 0.0);
        double stddev = std::sqrt(std::abs(sq_sum / stress_list.size() - mean * mean));
        std::cout << "Std. of Stress: " << stddev << std::endl;
        if (stddev > 1e-7)
            return -1;

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
