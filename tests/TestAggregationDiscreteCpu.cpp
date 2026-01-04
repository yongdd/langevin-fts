#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <string>
#include <array>
#include <chrono>
#include <numeric>

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
        // String to output file and print stream
        std::streamsize default_precision = std::cout.precision();

        // Temp
        double sum;
        double sum_total_partition;

        // -------------- initialize ------------
        // Platform type, [cuda, cpu-mkl]
        std::vector<int> nx = {64,64};
        std::vector<double> lx = {6.0,5.0};
        double ds = 1.0/10;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};

        std::vector<BlockInput> blocks =
        {
            {"A", 0.2, 0, 1},
            {"A", 0.2, 1, 2},
            {"A", 0.2, 2, 3},
            {"B", 0.4, 1, 4},
            {"B", 0.5, 2, 5},
            {"B", 0.6, 3, 6},
        };
        const int M = nx[0]*nx[1];

        //-------------- allocate array ------------
        double w      [2*M+nx.size()];
        double phi_a  [M];
        double phi_b  [M];

        // Choose platform
        std::string chain_model = "Discrete";
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<bool> aggregate_propagator_computations = {false, true};
        std::vector<bool> reduce_memory_usages = {false, true};
        std::vector<double> total_partition_list;
        for(std::string platform : avail_platforms)
        {
            if(platform != "cpu-mkl")
                continue;

            for(bool reduce_memory_usage : reduce_memory_usages)
            {
            for(bool aggregate_propagator_computation : aggregate_propagator_computations)
            {
                AbstractFactory<double> *factory = PlatformSelector::create_factory_real(platform, reduce_memory_usage);
                // factory->display_info();

                // Create instances and assign to the variables of base classes for the dynamic binding
                ComputationBox<double>* cb = factory->create_computation_box(nx, lx, {});
                Molecules* molecules       = factory->create_molecules_information(chain_model, ds, bond_lengths);
                molecules->add_polymer(1.0, blocks, {});
                PropagatorComputationOptimizer* propagator_computation_optimizer= new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
                PropagatorComputation<double>* solver     = factory->create_pseudospectral_solver(cb, molecules, propagator_computation_optimizer);

                // -------------- Print simulation parameters ------------
                std::cout << std::setprecision(default_precision);
                std::cout << std::boolalpha;
                std::cout << std::endl << "Chain Model: " << molecules->get_model_name() << std::endl;
                std::cout << "Platform: " << platform << std::endl;
                std::cout << "Using Aggregation: " << aggregate_propagator_computation << std::endl;
                std::cout << "Reduce Memory Usage: " << reduce_memory_usage << std::endl;

                // Display branches
                #ifndef NDEBUG
                propagator_computation_optimizer->display_blocks();
                propagator_computation_optimizer->display_propagators();
                #endif

                // std::cout<< "w_a and w_b are initialized to a zero." << std::endl;
                for(int i=0; i<M; i++)
                {
                    w[i]   = 0.0;
                    w[i+M] = 0.0;
                }

                // Keep the level of field value
                cb->zero_mean(&w[0]);
                cb->zero_mean(&w[M]);

                // For the given fields find the polymer statistics
                solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}}, {});
                solver->compute_concentrations();
                solver->get_total_concentration("A", phi_a);
                solver->get_total_concentration("B", phi_b);

                if (solver->check_total_partition() == false)
                    return -1;

                // Print iteration # and error levels and check the mass conservation
                sum = (cb->integral(phi_a) + cb->integral(phi_b))/cb->get_volume() - 1.0;
                std::cout<< "Mass error, total partitions" << std::endl;
                std::cout<< std::setw(13) << std::setprecision(3) << std::scientific << sum ;
                std::cout<< "\t[" << std::setprecision(7) << std::scientific << solver->get_total_partition(0);
                sum_total_partition = 0.0;
                for(int p=1; p<molecules->get_n_polymer_types(); p++)
                {
                    sum_total_partition += solver->get_total_partition(p);
                    std::cout<< std::setw(17) << std::setprecision(7) << std::scientific << solver->get_total_partition(p);
                }
                std::cout<< "]" << std::endl;

                total_partition_list.push_back(sum_total_partition);
                
                delete factory;
                delete cb;
                delete molecules;
                delete propagator_computation_optimizer;
                delete solver;
            }
            }
            double mean = std::accumulate(total_partition_list.begin(), total_partition_list.end(), 0.0)/total_partition_list.size();
            double sq_sum = std::inner_product(total_partition_list.begin(), total_partition_list.end(), total_partition_list.begin(), 0.0);
            double stddev = std::sqrt( std::abs(sq_sum / total_partition_list.size() - mean * mean ));
            std::cout << "Std. of sum_total_partition: " << stddev << std::endl;
            if (stddev > 1e-7)
                return -1;
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}