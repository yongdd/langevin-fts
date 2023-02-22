#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <array>
#include <chrono>

#include "Exception.h"
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

        double energy_total;
        // error_level = variable to check convergence of the iteration
        double error_level, old_error_level;

        // string to output file and print stream
        std::streamsize default_precision = std::cout.precision();

        // temp
        int idx;
        double sum;

        // -------------- initialize ------------
        // platform type, [cuda, cpu-mkl]
        
        int max_scft_iter = 3;
        double tolerance = 1e-9;

        double f = 0.2;
        double chi_n = 15.0;
        std::vector<int> nx = {16,16,16};
        std::vector<double> lx = {2.9,2.9,2.9};
        std::vector<double> lx_backup = lx;
        double ds = 1.0/10;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};

        std::vector<std::string> block_species = {"A","A","A","A","A","A","A","A", "B","B","B", "B","B","B","B", "B","B","B", "A","A","A"};
        std::vector<double> contour_lengths    = {  f,  f,  f,  f,  f,  f,  f,  f, 0.5,0.4,0.8, 0.3,0.4,0.1,0.5, 0.8,0.8,0.8,   f,  f,  f};
        std::vector<int> v                     = {  0,  1,  2,  3,  4,  5,  6,  7,   1,  2,  3,   4,  5,  6,  7,   9, 10, 11,   9, 10, 11};
        std::vector<int> u                     = {  1,  2,  3,  4,  5,  6,  7,  8,   9, 10, 11,  12, 13, 14, 15,  16, 17, 18,  19, 20, 21};

        // std::vector<std::string> block_species = {"A","A","A","A","A", "B","B","B","B", "B","B", "A","A"};
        // std::vector<double> contour_lengths = {f,f,f,f,f, 1-f,1-f,(1-f)/2,(1-f)/4, 1-f,1-f, f,f};
        // std::vector<int> v = {0,1,2,3,4, 1,2,3,4, 6,7, 6,7};
        // std::vector<int> u = {1,2,3,4,5, 6,7,8,9, 10,12, 11,13};

        // std::vector<std::string> block_species = {"A","A","A","A","A", "B","B", "B","B", "A","A"};
        // std::vector<double> contour_lengths = {f,f,f,f,f, 0.8,0.4, 1-f,1-f, f,f};
        // std::vector<int> v = {0,1,2,3,4, 1,2, 6,7, 6,7};
        // std::vector<int> u = {1,2,3,4,5, 6,7, 10,12, 11,13};

        const int M = nx[0]*nx[1]*nx[2];

        int am_n_var = 2*M+nx.size();  // A and B
        int am_max_hist = 20;
        double am_start_error = 1e-2;
        double am_mix_min = 0.02;
        double am_mix_init = 0.02;

        //-------------- allocate array ------------
        double w      [2*M+nx.size()];
        double w_out  [2*M+nx.size()];
        double w_diff [2*M+nx.size()];
        double xi     [M];
        double phi_a  [M];
        double phi_b  [M];
        double w_plus [M];
        double w_minus[M];

        // choose platform
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<bool> use_superpositions = {false, true};
        for(std::string chain_model : chain_models)
        {
            std::vector<double> energy_total_list;
            for(std::string platform : avail_platforms)
            {
                for(bool use_superposition : use_superpositions)
                {
                    AbstractFactory *factory = PlatformSelector::create_factory(platform, chain_model);
                    // factory->display_info();

                    // create instances and assign to the variables of base classes for the dynamic binding
                    ComputationBox *cb = factory->create_computation_box(nx, lx_backup);
                    Mixture* mx        = factory->create_mixture(ds, bond_lengths, use_superposition);
                    mx->add_polymer(1.0, block_species, contour_lengths, v, u, {});
                    Pseudo *pseudo     = factory->create_pseudo(cb, mx);
                    AndersonMixing *am = factory->create_anderson_mixing(am_n_var,
                                        am_max_hist, am_start_error, am_mix_min, am_mix_init);

                    // -------------- print simulation parameters ------------
                    std::cout << std::setprecision(default_precision);
                    std::cout << "Chain Model: " << mx->get_model_name() << std::endl;
                    std::cout << "Platform: " << platform << std::endl;
                    std::cout << "Using Superposition: " << use_superposition << std::endl;

                    // // display branches
                    // mx->display_unique_blocks();
                    // mx->display_unique_branches();

                    // std::cout<< "w_a and w_b are initialized to a gyroid." << std::endl;
                    double xx, yy, zz, c1, c2;
                    for(int i=0; i<nx[0]; i++)
                    {
                        xx = (i+1)*2*PI/nx[0];
                        for(int j=0; j<nx[1]; j++)
                        {
                            yy = (j+1)*2*PI/nx[1];
                            for(int k=0; k<nx[2]; k++)
                            {
                                zz = (k+1)*2*PI/nx[2];
                                c1 = sqrt(8.0/3.0)*(cos(xx)*sin(yy)*sin(2.0*zz) +
                                    cos(yy)*sin(zz)*sin(2.0*xx)+cos(zz)*sin(xx)*sin(2.0*yy));
                                c2 = sqrt(4.0/3.0)*(cos(2.0*xx)*cos(2.0*yy)+
                                    cos(2.0*yy)*cos(2.0*zz)+cos(2.0*zz)*cos(2.0*xx));
                                idx = i*nx[1]*nx[2] + j*nx[2] + k;
                                w[idx] = -0.3164*c1 +0.1074*c2;
                                w[idx+M] = 0.3164*c1 -0.1074*c2;
                            }
                        }
                    }

                    // keep the level of field value
                    cb->zero_mean(&w[0]);
                    cb->zero_mean(&w[M]);

                    // assign large initial value for the energy and error
                    energy_total = 1.0e20;
                    error_level = 1.0e20;

                    //------------------ run ----------------------
                    // iteration begins here
                    for(int iter=0; iter<max_scft_iter; iter++)
                    {
                        // for the given fields find the polymer statistics
                        pseudo->compute_statistics({}, {{"A",&w[0]},{"B",&w[M]}});
                        pseudo->get_monomer_concentration("A", phi_a);
                        pseudo->get_monomer_concentration("B", phi_b);

                        // compute stress
                        std::vector<double> stress = pseudo->compute_stress();

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
                            energy_total -= pc.get_volume_fraction()/pc.get_alpha()*log(pseudo->get_total_partition(p));
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
                        error_level += sqrt(stress[0]*stress[0] + stress[1]*stress[1] + stress[2]*stress[2]);

                        // print iteration # and error levels and check the mass conservation
                        sum = (cb->integral(phi_a) + cb->integral(phi_b))/cb->get_volume() - 1.0;
                        std::cout<< std::setw(8) << iter;
                        std::cout<< std::setw(13) << std::setprecision(3) << std::scientific << sum ;
                        std::cout<< "\t[" << std::setprecision(7) << std::scientific << pseudo->get_total_partition(0);
                        for(int p=1; p<mx->get_n_polymers(); p++)
                            std::cout<< std::setw(17) << std::setprecision(7) << std::scientific << pseudo->get_total_partition(p);
                        std::cout<< "]"; 
                        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << energy_total;
                        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << error_level << std::endl;

                        // std::cout<< " [";
                        // std::cout<< std::setw(10) << std::setprecision(7) << lx[0] << ", " << lx[1] << ", " << lx[2];
                        // std::cout<< "]" << std::endl;

                        // std::cout<< " [";
                        // std::cout<< std::setw(10) << std::setprecision(7) << stress[0] << ", " << stress[1] << ", " << stress[2];
                        // std::cout<< "]" << std::endl;

                        // conditions to end the iteration
                        if(error_level < tolerance) break;

                        // calculate new fields using simple and Anderson mixing  //w_new, w_current, w_diff
                        for(int d=0; d<cb->get_dim(); d++)
                        {
                            w[2*M+d] = cb->get_lx(d);
                            w_diff[2*M+d] = -stress[d];
                        }
                        am->calculate_new_fields(w, w, w_diff, old_error_level, error_level);

                        // update lx
                        for(int d=0; d<cb->get_dim(); d++)
                            lx[d] = w[2*M+d];
                        
                        cb->set_lx(lx);
                        pseudo->update_bond_function();
                    }
                    energy_total_list.push_back(energy_total);
                    
                    delete factory;
                    delete cb;
                    delete mx;
                    delete pseudo;
                    delete am;
                }
            }
            double mean = std::accumulate(energy_total_list.begin(), energy_total_list.end(), 0.0)/energy_total_list.size();
            double sq_sum = std::inner_product(energy_total_list.begin(), energy_total_list.end(), energy_total_list.begin(), 0.0);
            double stddev = std::sqrt( std::abs(sq_sum / energy_total_list.size() - mean * mean ));
            std::cout << "Std. of energy_level: " << stddev << std::endl;
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