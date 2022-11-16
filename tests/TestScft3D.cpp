#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <array>
#include <chrono>
#include <vector>
#include <algorithm>

#include "Exception.h"
#include "ParamParser.h"
#include "PolymerChain.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

int main()
{
    try
    {
        // math constatns
        const double PI = 3.14159265358979323846;
        // chrono timer
        std::chrono::system_clock::time_point chrono_start, chrono_end;
        std::chrono::duration<double> time_duration;

        // QQ = total partition function
        double QQ, energy_total;
        // error_level = variable to check convergence of the iteration
        double error_level, old_error_level;
        // input and output fields, xi is temporary storage for pressures
        double *w, *w_out, *w_diff;  // n_comp * MM
        double *xi, *w_plus, *w_minus; // MM
        // initial value of q, q_dagger
        double *q1_init, *q2_init;
        // segment concentration
        double *phi, *phia, *phib, *phitot;

        // string to output file and print stream
        std::ofstream print_stream;
        std::stringstream ss;
        std::string print_file_name;
        // temp
        int idx;
        double sum;

        // -------------- initialize ------------
        // platform type, [cuda, cpu-mkl]

        int max_scft_iter = 20;
        double tolerance = 1e-9;

        double f = 0.3;
        int n_segment = 50;
        double chi_n = 20.0;
        
        std::vector<int> nx = {31,49,23};
        std::vector<double> lx = {4.0,3.0,2.0};
        std::string chain_model = "Continuous";  // choose among [Continuous, Discrete]
        std::vector<int> N_chain = {int(std::round(f*n_segment)), int(std::round((1-f)*n_segment))};
        std::vector<double> bond_length = {1.0,1.0};
        double ds = 1.0/n_segment;

        int am_n_var = 2*nx[0]*nx[1]*nx[2];  // A and B
        int am_max_hist= 20;
        double am_start_error = 8e-1;
        double am_mix_min = 0.1;
        double am_mix_init = 0.1;

        // choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        for(std::string platform : avail_platforms){
            AbstractFactory *factory = PlatformSelector::create_factory(platform,chain_model);
            factory->display_info();

            // create instances and assign to the variables of base classs for the dynamic binding
            ComputationBox *cb  = factory->create_computation_box(nx, lx);
            PolymerChain *pc    = factory->create_polymer_chain(N_chain,bond_length,ds);
            Pseudo *pseudo      = factory->create_pseudo(cb, pc);
            AndersonMixing *am  = factory->create_anderson_mixing(am_n_var,
                                am_max_hist, am_start_error, am_mix_min, am_mix_init);

            // -------------- print simulation parameters ------------
            std::cout<< "---------- Simulation Parameters ----------" << std::endl;
            std::cout << "Box Dimension: " << cb->get_dim() << std::endl;
            std::cout << "chi_n, f, N: " << chi_n << " " << f << " " << pc->get_n_segment_total() << std::endl;
            std::cout << "Nx: " << cb->get_nx(0) << " " << cb->get_nx(1) << " " << cb->get_nx(2) << std::endl;
            std::cout << "Lx: " << cb->get_lx(0) << " " << cb->get_lx(1) << " " << cb->get_lx(2) << std::endl;
            std::cout << "dx: " << cb->get_dx(0) << " " << cb->get_dx(1) << " " << cb->get_dx(2) << std::endl;
            sum = 0.0;
            for(int i=0; i<cb->get_n_grid(); i++)
                sum += cb->get_dv(i);
            std::cout << "volume, sum(dv):  " << cb->get_volume() << " " << sum << std::endl;

            //-------------- allocate array ------------
            w       = new double[cb->get_n_grid()*2];
            w_out   = new double[cb->get_n_grid()*2];
            w_diff  = new double[cb->get_n_grid()*2];
            xi      = new double[cb->get_n_grid()];
            phi     = new double[cb->get_n_grid()*2];
            phitot  = new double[cb->get_n_grid()];
            w_plus  = new double[cb->get_n_grid()];
            w_minus = new double[cb->get_n_grid()];
            q1_init = new double[cb->get_n_grid()];
            q2_init = new double[cb->get_n_grid()];

            phia = &phi[0];
            phib = &phi[cb->get_n_grid()];
            //-------------- setup fields ------------
            //call random_number(phia)
            //   phia = reshape( phia, (/ x_hi-x_lo+1,y_hi-y_lo+1,z_hi-z_lo+1 /), order = (/ 3, 2, 1 /))
            //   call random_number(phia(:,:,z_lo))
            //   do k=z_lo,z_hi
            //     phia(:,:,k) = phia(:,:,z_lo)
            //   end do

            std::cout<< "w_a and w_b are initialized to a given test fields." << std::endl;
            for(int i=0; i<cb->get_nx(0); i++)
                for(int j=0; j<cb->get_nx(1); j++)
                    for(int k=0; k<cb->get_nx(2); k++)
                    {
                        idx = i*cb->get_nx(1)*cb->get_nx(2) + j*cb->get_nx(2) + k;
                        phia[idx]= cos(2.0*PI*i/4.68)*cos(2.0*PI*j/3.48)*cos(2.0*PI*k/2.74)*0.1;
                    }

            for(int i=0; i<cb->get_n_grid(); i++)
            {
                phib[i] = 1.0 - phia[i];
                w[i]              = chi_n*phib[i];
                w[i+cb->get_n_grid()] = chi_n*phia[i];
            }

            // keep the level of field value
            cb->zero_mean(&w[0]);
            cb->zero_mean(&w[cb->get_n_grid()]);

            // free end initial condition. q1 is q and q2 is qdagger.
            // q1 starts from A end and q2 starts from B end.
            for(int i=0; i<cb->get_n_grid(); i++)
            {
                q1_init[i] = 1.0;
                q2_init[i] = 1.0;
            }
            //print_stream->close();

            // assign large initial value for the energy and error
            energy_total = 1.0e20;
            error_level = 1.0e20;

            //------------------ run ----------------------
            std::cout<< "---------- Run ----------" << std::endl;
            std::cout<< "iteration, mass error, total_partition, energy_total, error_level" << std::endl;
            chrono_start = std::chrono::system_clock::now();
            // iteration begins here
            for(int iter=0; iter<max_scft_iter; iter++)
            {
                // for the given fields find the polymer statistics
                pseudo->compute_statistics(phi,q1_init,q2_init,w,QQ);

                // calculate the total energy
                for(int i=0; i<cb->get_n_grid(); i++)
                {
                    w_minus[i] = (w[i]-w[i + cb->get_n_grid()])/2;
                    w_plus[i]  = (w[i]+w[i + cb->get_n_grid()])/2;
                }
                energy_total  = -log(QQ/cb->get_volume());
                energy_total += cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                energy_total += cb->integral(w_plus)/cb->get_volume();
                //energy_total += cb->inner_product(ext_w_minus,ext_w_minus)/chi_b/cb->get_volume();

                for(int i=0; i<cb->get_n_grid(); i++)
                {
                    // calculate pressure field for the new field calculation, the method is modified from Fredrickson's
                    xi[i] = 0.5*(w[i]+w[i+cb->get_n_grid()]-chi_n);
                    // calculate output fields
                    w_out[i]              = chi_n*phib[i] + xi[i];
                    w_out[i+cb->get_n_grid()] = chi_n*phia[i] + xi[i];
                }
                cb->zero_mean(&w_out[0]);
                cb->zero_mean(&w_out[cb->get_n_grid()]);

                // error_level measures the "relative distance" between the input and output fields
                old_error_level = error_level;
                for(int i=0; i<2*cb->get_n_grid(); i++)
                    w_diff[i] = w_out[i]- w[i];
                error_level = sqrt(cb->multi_inner_product(2,w_diff,w_diff)/
                                (cb->multi_inner_product(2,w,w)+1.0));

                // print iteration # and error levels and check the mass conservation
                sum = (cb->integral(phia) + cb->integral(phib))/cb->get_volume() - 1.0;
                std::cout<< std::setw(8) << iter;
                std::cout<< std::setw(13) << std::setprecision(3) << std::scientific << sum ;
                std::cout<< std::setw(17) << std::setprecision(7) << std::scientific << QQ;
                std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << energy_total;
                std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << error_level << std::endl;

                // conditions to end the iteration
                if(error_level < tolerance) break;
                // calculte new fields using simple and Anderson mixing
                am->calculate_new_fields(w, w_out, w_diff, old_error_level, error_level);
            }

            // estimate execution time
            chrono_end = std::chrono::system_clock::now();
            time_duration = chrono_end - chrono_start;
            std::cout<< "total time, time per step: ";
            std::cout<< time_duration.count() << " " << time_duration.count()/max_scft_iter << std::endl;

            //------------- finalize -------------
            delete[] w;
            delete[] w_out;
            delete[] w_diff;
            delete[] xi;
            delete[] phi;
            delete[] phitot;
            delete[] w_plus;
            delete[] w_minus;
            delete[] q1_init;
            delete[] q2_init;

            delete pc;
            delete cb;
            delete pseudo;
            delete am;
            delete factory;

            if (std::isnan(error_level) || std::abs(error_level-0.009307729) > 1e-7)
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
