#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <array>
#include <chrono>

#include "Exception.h"
#include "ParamParser.h"
#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

int main()
{
    try
    {
        // math constants
        const double PI = 3.14159265358979323846;
        // chrono timer
        std::chrono::system_clock::time_point chrono_start, chrono_end;
        std::chrono::duration<double> time_duration;

        double energy_total;
        // error_level = variable to check convergence of the iteration
        double error_level, old_error_level;
        // input and output fields, xi is temporary storage for pressures
        double *w, *w_out, *w_diff;  // n_comp * MM
        double *xi, *w_plus, *w_minus; // MM
        // segment concentration
        double *phi_a, *phi_b, *phi_tot;

        // string to output file and print stream
        std::streamsize default_precision = std::cout.precision();
        std::ofstream print_stream;
        std::stringstream ss;
        std::string print_file_name;
        // temp
        int idx;
        double sum;

        double old_lx;
        // double new_lx;
        // double new_dlx;
        // -------------- initialize ------------
        // platform type, [cuda, cpu-mkl]
        
        int max_scft_iter = 200;
        double tolerance = 1e-9;

        double f = 0.5;
        double chi_n = 10.0;
        std::vector<int> nx = {256};
        std::vector<double> lx = {1.5};
        double ds = 1.0/100;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.5}};
        std::vector<std::string> block_species = {"A","A","B","B","A","A"};
        std::vector<double> contour_lengths = {0.6,1.2,1.2,0.9,0.9,1.2};
        std::vector<int> v = {0,0,0,0,1,1};
        std::vector<int> u = {1,2,5,6,4,15};

        int am_n_var = 2*nx[0];  // A and B
        int am_max_hist = 20;
        double am_start_error = 8e-1;
        double am_mix_min = 0.1;
        double am_mix_init = 0.1;

        const int M = nx[0];
        //-------------- allocate array ------------
        w       = new double[2*M];
        w_out   = new double[2*M];
        w_diff  = new double[2*M];
        xi      = new double[M];
        phi_a   = new double[M];
        phi_b   = new double[M];
        phi_tot = new double[M];
        w_plus  = new double[M];
        w_minus = new double[M];

        // choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        for(std::string platform : avail_platforms)
        {
            for(std::string chain_model : chain_models)
            {
                AbstractFactory *factory = PlatformSelector::create_factory(platform, chain_model);
                factory->display_info();

                // create instances and assign to the variables of base classes for the dynamic binding
                ComputationBox *cb = factory->create_computation_box(nx, lx);
                Mixture* mx        = factory->create_mixture(ds, bond_lengths);
                mx->add_polymer(1.0, block_species, contour_lengths, v, u, {});
                Pseudo *pseudo     = factory->create_pseudo(cb, mx);
                AndersonMixing *am = factory->create_anderson_mixing(am_n_var,
                                    am_max_hist, am_start_error, am_mix_min, am_mix_init);

                // -------------- print simulation parameters ------------
                std::cout << std::setprecision(default_precision);
                // std::cout<< "---------- Simulation Parameters ----------" << std::endl;
                // std::cout << "Box Dimension: " << cb->get_dim() << std::endl;
                std::cout << "Chain Model: " << mx->get_model_name() << std::endl;
                // std::cout << "chi_n, f: " << chi_n << " " << f << " "  << std::endl;
                // std::cout << "Nx: " << cb->get_nx(0) << " " << cb->get_nx(1) << " " << cb->get_nx(2) << std::endl;
                // std::cout << "Lx: " << cb->get_lx(0) << " " << cb->get_lx(1) << " " << cb->get_lx(2) << std::endl;
                // std::cout << "dx: " << cb->get_dx(0) << " " << cb->get_dx(1) << " " << cb->get_dx(2) << std::endl;

                // std::cout<< "w_a and w_b are initialized to a lamellar." << std::endl;
                for(int i=0; i<cb->get_nx(0); i++)
                    for(int j=0; j<cb->get_nx(1); j++)
                        for(int k=0; k<cb->get_nx(2); k++)
                        {
                            w[k]   =  cos(2.0*PI*k/cb->get_nx(2))*10;
                            w[k+M] = -cos(2.0*PI*k/cb->get_nx(2))*10;
                        }

                // keep the level of field value
                cb->zero_mean(&w[0]);
                cb->zero_mean(&w[M]);

                // assign large initial value for the energy and error
                energy_total = 1.0e20;
                error_level = 1.0e20;

                //------------------ run ----------------------
                // std::cout<< "---------- Run ----------" << std::endl;
                // std::cout<< "iteration, mass error, total_partitions, energy_total, error_level" << std::endl;
                chrono_start = std::chrono::system_clock::now();
                // iteration begins here
                for(int iter=0; iter<max_scft_iter; iter++)
                {
                    // for the given fields find the polymer statistics
                    pseudo->compute_statistics({}, {{"A",&w[0]},{"B",&w[M]}});
                    pseudo->get_species_concentration("A", phi_a);
                    pseudo->get_species_concentration("B", phi_b);

                    // calculate the total energy
                    for(int i=0; i<M; i++)
                    {
                        w_minus[i] = (w[i]-w[i+M])/2;
                        w_plus[i]  = (w[i]+w[i+M])/2;
                    }

                    energy_total = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                    energy_total -= cb->integral(w_plus)/cb->get_volume();
                    for(int p=0; p<mx->get_n_polymers(); p++){
                        PolymerChain& pc = mx->get_polymer(p);
                        energy_total -= pc.get_volume_fraction()/pc.get_alpha()*log(pseudo->get_total_partition(p)/cb->get_volume());
                    }

                    for(int i=0; i<M; i++)
                    {
                        // calculate pressure field for the new field calculation
                        xi[i] = 0.5*(w[i]+w[i+M]-chi_n);
                        // calculate output fields
                        w_out[i]   = chi_n*phi_b[i] + xi[i];
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

                    // print iteration # and error levels and check the mass conservation
                    sum = (cb->integral(phi_a) + cb->integral(phi_b))/cb->get_volume() - 1.0;

                    // conditions to end the iteration
                    if(error_level < tolerance) break;

                    // calculate new fields using simple and Anderson mixing
                    am->calculate_new_fields(w, w_out, w_diff, old_error_level, error_level);
                }

                double dL = 0.0000001;
                double old_lx = lx[0];
                //----------- compute derivate of H: lx + delta ----------------
                lx[0] = old_lx + dL/2;
                cb->set_lx(lx);
                pseudo->update();

                // for the given fields find the polymer statistics
                pseudo->compute_statistics({}, {{"A",&w[0]},{"B",&w[M]}});
                pseudo->get_species_concentration("A", phi_a);
                pseudo->get_species_concentration("B", phi_b);

                // calculate the total energy
                for(int i=0; i<M; i++)
                {
                    w_minus[i] = (w[i]-w[i+M])/2;
                    w_plus[i]  = (w[i]+w[i+M])/2;
                }

                double energy_total_1 = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                energy_total_1 -= cb->integral(w_plus)/cb->get_volume();
                for(int p=0; p<mx->get_n_polymers(); p++){
                    PolymerChain& pc = mx->get_polymer(p);
                    energy_total_1 -= pc.get_volume_fraction()/pc.get_alpha()*log(pseudo->get_total_partition(p)/cb->get_volume());
                }


                //----------- compute derivate of H: lx - delta ----------------
                lx[0] = old_lx - dL/2;
                cb->set_lx(lx);
                pseudo->update();

                // for the given fields find the polymer statistics
                pseudo->compute_statistics({}, {{"A",&w[0]},{"B",&w[M]}});
                pseudo->get_species_concentration("A", phi_a);
                pseudo->get_species_concentration("B", phi_b);

                // calculate the total energy
                for(int i=0; i<M; i++)
                {
                    w_minus[i] = (w[i]-w[i+M])/2;
                    w_plus[i]  = (w[i]+w[i+M])/2;
                }

                double energy_total_2 = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                energy_total_2 -= cb->integral(w_plus)/cb->get_volume();
                for(int p=0; p<mx->get_n_polymers(); p++){
                    PolymerChain& pc = mx->get_polymer(p);
                    energy_total_2 -= pc.get_volume_fraction()/pc.get_alpha()*log(pseudo->get_total_partition(p)/cb->get_volume());
                }

                // compute stress
                double dh_dl = (energy_total_1-energy_total_2)/dL;
                auto stress = pseudo->compute_stress();
                std:: cout << "dH/dL : " << dh_dl << std::endl;
                std:: cout << "Stress : " << stress[2] << std::endl;
                double relative_stress_error = std::abs(dh_dl-stress[2])/std::abs(stress[2]);
                std:: cout << "Relative stress error : " << relative_stress_error << std::endl;
                if (!std::isfinite(relative_stress_error) || std::abs(relative_stress_error) > 1e-3)
                    return -1;

                //------------- finalize -------------
                delete mx;
                delete cb;
                delete pseudo;
                delete am;
                delete factory;
            }
        }
        delete[] w;
        delete[] w_out;
        delete[] w_diff;
        delete[] xi;

        delete[] phi_a;
        delete[] phi_b;
        delete[] phi_tot;
        delete[] w_plus;
        delete[] w_minus;

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
