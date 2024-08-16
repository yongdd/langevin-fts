#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <array>
#include <chrono>

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
        // Chrono timer
        std::chrono::system_clock::time_point chrono_start, chrono_end;
        std::chrono::duration<double> time_duration;

        double energy_total;
        // error_level = variable to check convergence of the iteration
        double error_level, old_error_level;
        // Input and output fields, xi is temporary storage for pressures
        double *w, *w_out, *w_diff;  // n_comp * M
        double *xi, *w_plus, *w_minus; // M
        // Segment concentration
        double *phi_a, *phi_b, *phi_tot;

        // String to output file and print stream
        std::streamsize default_precision = std::cout.precision();
        std::ofstream print_stream;
        std::stringstream ss;
        std::string print_file_name;
        // Temp
        int idx;
        double sum;

        // -------------- Initialize ------------
        // Platform type, [cuda, cpu-mkl]
        
        int max_scft_iter = 20;
        double tolerance = 1e-9;

        double f = 0.3;
        double chi_n = 25.0;
        std::vector<int> nx = {263};
        std::vector<double> lx = {4.0};
        std::string chain_model = "Continuous";  // choose among [Continuous, Discrete]
        double ds = 1.0/50;

        int am_n_var = 2*nx[0];  // A and B
        int am_max_hist = 20;
        double am_start_error = 8e-1;
        double am_mix_min = 0.1;
        double am_mix_init = 0.1;

        std::vector<BlockInput> blocks =
        {
            {"A",    f, 0, 1},
            {"B",1.0-f, 1, 2},
        };

        bool reduce_memory_usage=false;

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        for(std::string platform : avail_platforms){
            AbstractFactory *factory = PlatformSelector::create_factory(platform, reduce_memory_usage);
            factory->display_info();

            // Create instances and assign to the variables of base classes for the dynamic binding
            ComputationBox *cb = factory->create_computation_box(nx, lx, {});
            Molecules* molecules        = factory->create_molecules_information(chain_model, ds, {{"A",1.0}, {"B",1.0}});
            molecules->add_polymer(1.0, blocks, {});
            PropagatorAnalyzer* propagator_analyzer= new PropagatorAnalyzer(molecules, false);
            PropagatorComputation *solver     = factory->create_pseudospectral_solver(cb, molecules, propagator_analyzer);
            AndersonMixing *am = factory->create_anderson_mixing(am_n_var,
                                am_max_hist, am_start_error, am_mix_min, am_mix_init);

            const int M = cb->get_n_grid();

            // -------------- Print simulation parameters ------------
            std::cout << std::setprecision(default_precision);
            std::cout << std::boolalpha;
            std::cout<< "---------- Simulation Parameters ----------" << std::endl;
            std::cout << "Box Dimension: " << cb->get_dim() << std::endl;
            std::cout << "Chain Model: " << molecules->get_model_name() << std::endl;
            std::cout << "chi_n, f: " << chi_n << " " << f << " "  << std::endl;
            std::cout << "Nx: " << cb->get_nx(0) << std::endl;
            std::cout << "Lx: " << cb->get_lx(0) << std::endl;
            std::cout << "dx: " << cb->get_dx(0) << std::endl;
            sum = 0.0;
            for(int i=0; i<M; i++)
                sum += cb->get_dv(i);
            std::cout << "volume, sum(dv):  " << cb->get_volume() << " " << sum << std::endl;

            propagator_analyzer->display_blocks();
            propagator_analyzer->display_propagators();

            //-------------- Allocate array ------------
            w       = new double[M*2];
            w_out   = new double[M*2];
            w_diff  = new double[M*2];
            xi      = new double[M];
            phi_a   = new double[M];
            phi_b   = new double[M];
            phi_tot = new double[M];
            w_plus  = new double[M];
            w_minus = new double[M];
            //-------------- Setup fields ------------
            //call random_number(phi_a)
            //   phi_a = reshape( phi_a, (/ x_hi-x_lo+1,y_hi-y_lo+1,z_hi-z_lo+1 /), order = (/ 3, 2, 1 /))
            //   call random_number(phi_a(:,:,z_lo))
            //   do k=z_lo,z_hi
            //     phi_a(:,:,k) = phi_a(:,:,z_lo)
            //   end do

            std::cout<< "w_a and w_b are initialized to a given test fields." << std::endl;
            for(int k=0; k<cb->get_nx(0); k++)
            {
                idx = k;
                phi_a[idx]= cos(2.0*PI*0/4.68)*cos(2.0*PI*0/3.48)*cos(2.0*PI*k/2.74)*0.1;
            }

            for(int i=0; i<M; i++)
            {
                phi_b[i] = 1.0 - phi_a[i];
                w[i]   = chi_n*phi_b[i];
                w[i+M] = chi_n*phi_a[i];
            }

            // Keep the level of field value
            cb->zero_mean(&w[0]);
            cb->zero_mean(&w[M]);

            // Assign large initial value for the energy and error
            energy_total = 1.0e20;
            error_level = 1.0e20;

            //------------------ run ----------------------
            std::cout<< "---------- Run ----------" << std::endl;
            std::cout<< "iteration, mass error, total partitions, total energy, error level" << std::endl;
            chrono_start = std::chrono::system_clock::now();
            // Iteration begins here
            for(int iter=0; iter<max_scft_iter; iter++)
            {
                // For the given fields find the polymer statistics
                solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                solver->compute_concentrations();
                solver->get_total_concentration("A", phi_a);
                solver->get_total_concentration("B", phi_b);

                // Calculate the total energy
                for(int i=0; i<M; i++)
                {
                    w_minus[i] = (w[i]-w[i + M])/2;
                    w_plus[i]  = (w[i]+w[i + M])/2;
                }

                energy_total = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                energy_total -= cb->integral(w_plus)/cb->get_volume();
                // energy_total += cb->inner_product(ext_w_minus,ext_w_minus)/chi_b/cb->get_volume();
                for(int p=0; p<molecules->get_n_polymer_types(); p++){
                    Polymer& pc = molecules->get_polymer(p);
                    energy_total -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
                }

                for(int i=0; i<M; i++)
                {
                    // Calculate pressure field for the new field calculation
                    xi[i] = 0.5*(w[i]+w[i+M]-chi_n);
                    // Calculate output fields
                    w_out[i]                  = chi_n*phi_b[i] + xi[i];
                    w_out[i+M] = chi_n*phi_a[i] + xi[i];
                }
                cb->zero_mean(&w_out[0]);
                cb->zero_mean(&w_out[M]);

                // error_level measures the "relative distance" between the input and output fields
                old_error_level = error_level;
                for(int i=0; i<2*M; i++)
                    w_diff[i] = w_out[i]- w[i];
                error_level = sqrt(cb->multi_inner_product(2,w_diff,w_diff)/
                                (cb->multi_inner_product(2,w,w)+1.0));

                // Print iteration # and error levels and check the mass conservation
                sum = (cb->integral(phi_a) + cb->integral(phi_b))/cb->get_volume() - 1.0;
                std::cout<< std::setw(8) << iter;
                std::cout<< std::setw(13) << std::setprecision(3) << std::scientific << sum ;
                std::cout<< "\t[" << std::setprecision(7) << std::scientific << solver->get_total_partition(0);
                for(int p=1; p<molecules->get_n_polymer_types(); p++)
                    std::cout<< std::setw(17) << std::setprecision(7) << std::scientific << solver->get_total_partition(p);
                std::cout<< "]"; 
                std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << energy_total;
                std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << error_level << std::endl;

                // Conditions to end the iteration
                if(error_level < tolerance) break;
                // Calculate new fields using simple and Anderson mixing
                                //w_new, w_current, w_diff
                am->calculate_new_fields(w, w, w_diff, old_error_level, error_level);
            }

            // Estimate execution time
            chrono_end = std::chrono::system_clock::now();
            time_duration = chrono_end - chrono_start;
            std::cout<< "total time, time per step: ";
            std::cout<< time_duration.count() << " " << time_duration.count()/max_scft_iter << std::endl;

            //------------- Finalize -------------
            delete[] w;
            delete[] w_out;
            delete[] w_diff;
            delete[] xi;

            delete[] phi_a;
            delete[] phi_b;
            delete[] phi_tot;
            delete[] w_plus;
            delete[] w_minus;

            delete molecules;
            delete propagator_analyzer;
            delete cb;
            delete solver;
            delete am;
            delete factory;

            if (!std::isfinite(error_level) || std::abs(error_level-0.018584567) > 1e-4)
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
