#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <array>
#include <chrono>
#include <numeric>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorAnalyzer.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

int main()
{
    try
    {
        // Math constants
        const double PI = 3.14159265358979323846;

        // String to output file and print stream
        std::streamsize default_precision = std::cout.precision();

        // Temp
        double sum;
        double sum_total_partition;

        // -------------- initialize ------------
        // Platform type, [cuda, cpu-mkl]
        double f = 0.2;
        std::vector<int> nx = {64,64};
        std::vector<double> lx = {6.0,5.0};
        std::vector<double> lx_backup = lx;
        double ds = 1.0/10;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};

        std::vector<BlockInput> blocks =
        {
            {"A",  f, 0, 1},
            {"A",  f, 1, 2},
            {"A",  f, 2, 3},
            {"A",  f, 3, 4},
            {"A",  f, 4, 5},
            {"A",  f, 5, 6},
            {"A",  f, 6, 7},
            {"A",  f, 7, 8},
            {"B", 0.5, 1, 9},
            {"B", 0.4, 2,10},
            {"B", 0.8, 3,11},
            {"B", 0.3, 4,12},
            {"B", 0.4, 5,13},
            {"B", 0.1, 6,14},
            {"B", 0.5, 7,15},
            {"B", 0.8, 9,16},
            {"B", 0.8,10,17},
            {"B", 0.8,11,18},
            {"A",   f, 9,19},
            {"A",   f,10,20},
            {"A",   f,11,21},
            {"B", 0.8, 9,22},
            {"B", 0.8,10,23},
            {"B", 0.8,11,24},
            {"B", 0.3, 4,25},
            {"B", 0.1, 6,27},
            {"B", 0.5, 7,28},
        };

        // std::vector<std::string> block_species = {"A","A","A","A","A", "B","B","B","B", "B","B", "A","A"};
        // std::vector<double> contour_lengths = {f,f,f,f,f, 1-f,1-f,(1-f)/2,(1-f)/4, 1-f,1-f, f,f};
        // std::vector<int> v = {0,1,2,3,4, 1,2,3,4, 6,7, 6,7};
        // std::vector<int> u = {1,2,3,4,5, 6,7,8,9, 10,12, 11,13};

        // std::vector<std::string> block_species = {"A","A","A","A","A", "B","B", "B","B", "A","A"};
        // std::vector<double> contour_lengths = {f,f,f,f,f, 0.8,0.4, 1-f,1-f, f,f};
        // std::vector<int> v = {0,1,2,3,4, 1,2, 6,7, 6,7};
        // std::vector<int> u = {1,2,3,4,5, 6,7, 10,12, 11,13};

        const int M = nx[0]*nx[1];

        int am_n_var = 2*M+nx.size();  // A and B
        int am_max_hist = 20;
        double am_start_error = 1e-2;
        double am_mix_min = 0.02;
        double am_mix_init = 0.02;

        //-------------- allocate array ------------
        double w      [2*M+nx.size()];
        double phi_a  [M];
        double phi_b  [M];
        double mask   [M];

        // Set a mask to set q(r,s) = 0
        double nano_particle_radius = 1.5;
        for(int i=0;i<nx[0];i++)
        {
            for(int j=0;j<nx[1];j++)
            {
                if (sqrt(pow(i*lx[0]/nx[0]-3.0,2) + pow(j*lx[1]/nx[1]-2.5,2)) < nano_particle_radius)
                    mask[i+j*nx[0]] = 0.0;
                else
                    mask[i+j*nx[0]] = 1.0;
            }
        }
        sum = 0.0;
        for(int i=0; i<M; i++)
             sum += mask[i]/M;
        // std::cout << "Mean of mask, volume fraction of sphere: " << 1.0 - sum << ", " << 4.0/3.0*PI*pow(nano_particle_radius,3)/(lx[0]*lx[1]) << std::endl;
        std::cout << "Mean of mask, volume fraction of disk: " << 1.0 - sum << ", " << PI*pow(nano_particle_radius,2)/(lx[0]*lx[1]) << std::endl;

        // Choose platform
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<bool> aggregate_propagator_computations = {false, true};
        std::vector<bool> reduce_memory_usages = {false, true};
        for(std::string chain_model : chain_models)
        {
            std::vector<double> total_partition_list;
            for(std::string platform : avail_platforms)
            {
                for(bool aggregate_propagator_computation : aggregate_propagator_computations)
                {
                    for(bool reduce_memory_usage : reduce_memory_usages)
                    {
                        // 'cpu-mkl' does not support reduce_gpu_memory_usage
                        if(reduce_memory_usage == true && platform == "cpu-mkl")
                                continue;
                        
                        AbstractFactory *factory = PlatformSelector::create_factory(platform, reduce_memory_usage);
                        // factory->display_info();

                        // Create instances and assign to the variables of base classes for the dynamic binding
                        ComputationBox *cb = factory->create_computation_box(nx, lx_backup, {}, mask);
                        Molecules* molecules        = factory->create_molecules_information(chain_model, ds, bond_lengths);
                        molecules->add_polymer(1.0, blocks, {});
                        PropagatorAnalyzer* propagator_analyzer= new PropagatorAnalyzer(molecules, aggregate_propagator_computation);
                        PropagatorComputation *solver     = factory->create_pseudospectral_solver(cb, molecules, propagator_analyzer);
                        AndersonMixing *am = factory->create_anderson_mixing(am_n_var,
                                            am_max_hist, am_start_error, am_mix_min, am_mix_init);

                        // -------------- Print simulation parameters ------------
                        std::cout << std::setprecision(default_precision);
                        std::cout << std::boolalpha;
                        std::cout << std::endl << "Chain Model: " << molecules->get_model_name() << std::endl;
                        std::cout << "Platform: " << platform << std::endl;
                        std::cout << "Using Aggregation: " << aggregate_propagator_computation << std::endl;
                        std::cout << "Reducing GPU Memory Usage: " << reduce_memory_usage << std::endl;

                        // // display branches
                        // PropagatorAnalyzer->display_blocks();
                        // PropagatorAnalyzer->display_propagators();

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
                        delete propagator_analyzer;
                        delete solver;
                        delete am;
                    }
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