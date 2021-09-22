
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <array>
#include <chrono>

#include "ParamParser.h"
#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

int main(int argc, char **argv)
{
    // math constatns
    const double PI = 3.14159265358979323846;
    // chrono timer
    std::chrono::system_clock::time_point chrono_start, chrono_end;
    std::chrono::duration<double> time_duration;
    // iter = number of iteration steps, maxiter = maximum number of iteration steps
    int iter;
    // QQ = total partition function
    double QQ, energy_chain, energy_field, energy_total, energy_old;
    // error_level = variable to check convergence of the iteration
    double error_level, old_error_level;
    // input and output fields, xi is temporary storage for pressures
    double *w, *w_out, *w_diff;  // n_comp * MM
    double *xi, *w_plus, *w_minus; // MM
    // initial value of q, q_dagger
    double *q1_init, *q2_init;
    // segment concentration
    double *phia, *phib, *phitot;

    // string to output file and print stream
    std::ofstream print_stream;
    std::stringstream ss;
    std::string print_file_name;
    // temp
    int idx;
    double sum;

    // -------------- initialize ------------
    // platform type, (CUDA, CPU_MKL or CPU_FFTW)
    std::string str_platform = "CUDA";
    
    int max_scft_iter = 20;
    double tolerance = 1e-9;

    double f = 0.3;
    int NN = 50;
    double chi_n = 20.0;
    std::vector<int> nx = {49, 63};
    std::vector<double> lx = {4.0,3.0};
    std::string polymer_model = "Gaussian";  // choose among [Gaussian, Discrete]

    int am_n_comp = 2;  // A and B
    int am_max_hist= 20;
    double am_start_error = 8e-1;
    double am_mix_min = 0.1;
    double am_mix_init = 0.1;

    // choose platform
    AbstractFactory *factory = PlatformSelector::create_factory(str_platform);

    // create instances and assign to the variables of base classs
    // for the dynamic binding
    PolymerChain *pc = factory->create_polymer_chain(f, NN, chi_n);
    SimulationBox *sb = factory->create_simulation_box(nx, lx);
    Pseudo *pseudo = factory->create_pseudo(sb, pc, polymer_model);
    AndersonMixing *am = factory->create_anderson_mixing(sb, am_n_comp,
                                        am_max_hist, am_start_error, am_mix_min, am_mix_init);

    // -------------- print simulation parameters ------------
    std::cout<< "---------- Simulation Parameters ----------" << std::endl;
    std::cout << "Box Dimension: " << sb->get_dimension() << std::endl;
    std::cout << "Precision: 8" << std::endl;
    std::cout << "chi_n, f, NN: " << pc->get_chi_n() << " " << pc->get_f() << " " << pc->get_NN() << std::endl;
    std::cout << "Nx: " << sb->get_nx(0) << " " << sb->get_nx(1) << " " << sb->get_nx(2) << std::endl;
    std::cout << "Lx: " << sb->get_lx(0) << " " << sb->get_lx(1) << " " << sb->get_lx(2) << std::endl;
    std::cout << "dx: " << sb->get_dx(0) << " " << sb->get_dx(1) << " " << sb->get_dx(2) << std::endl;
    sum = 0.0;
    for(int i=0; i<sb->get_MM(); i++)
        sum += sb->get_dv(i);
    std::cout << "volume, sum(dv):  " << sb->get_volume() << " " << sum << std::endl;

    //-------------- allocate array ------------
    w       = new double[sb->get_MM()*2];
    w_out   = new double[sb->get_MM()*2];
    w_diff  = new double[sb->get_MM()*2];
    xi      = new double[sb->get_MM()];
    phia    = new double[sb->get_MM()];
    phib    = new double[sb->get_MM()];
    phitot  = new double[sb->get_MM()];
    w_plus  = new double[sb->get_MM()];
    w_minus = new double[sb->get_MM()];
    q1_init = new double[sb->get_MM()];
    q2_init = new double[sb->get_MM()];

    //-------------- setup fields ------------
    //call random_number(phia)
    //   phia = reshape( phia, (/ x_hi-x_lo+1,y_hi-y_lo+1,z_hi-z_lo+1 /), order = (/ 3, 2, 1 /))
    //   call random_number(phia(:,:,z_lo))
    //   do k=z_lo,z_hi
    //     phia(:,:,k) = phia(:,:,z_lo)
    //   end do

    std::cout<< "wminus and wplus are initialized to a given test fields." << std::endl;
    for(int i=0; i<sb->get_nx(0); i++)
        for(int j=0; j<sb->get_nx(1); j++)
            for(int k=0; k<sb->get_nx(2); k++)
            {
                idx = i*sb->get_nx(1)*sb->get_nx(2) + j*sb->get_nx(2) + k;
                phia[idx]= cos(2.0*PI*i/4.68)*cos(2.0*PI*j/3.48)*cos(2.0*PI*k/2.74)*0.1;
            }

    for(int i=0; i<sb->get_MM(); i++)
    {
        phib[i] = 1.0 - phia[i];
        w[i]              = pc->get_chi_n()*phib[i];
        w[i+sb->get_MM()] = pc->get_chi_n()*phia[i];
    }

    // keep the level of field value
    sb->zero_mean(&w[0]);
    sb->zero_mean(&w[sb->get_MM()]);

    // free end initial condition. q1 is q and q2 is qdagger.
    // q1 starts from A end and q2 starts from B end.
    for(int i=0; i<sb->get_MM(); i++)
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
        pseudo->find_phi(phia, phib,q1_init,q2_init,&w[0],&w[sb->get_MM()],QQ);

        // calculate the total energy
        energy_old = energy_total;
        for(int i=0; i<sb->get_MM(); i++)
        {
            w_minus[i] = (w[i]-w[i + sb->get_MM()])/2;
            w_plus[i]  = (w[i]+w[i + sb->get_MM()])/2;
        }
        energy_total  = -log(QQ/sb->get_volume());
        energy_total += sb->inner_product(w_minus,w_minus)/pc->get_chi_n()/sb->get_volume();
        energy_total += sb->integral(w_plus)/sb->get_volume();
        //energy_total += sb->inner_product(ext_w_minus,ext_w_minus)/chi_b/sb->get_volume();

        for(int i=0; i<sb->get_MM(); i++)
        {
            // calculate pressure field for the new field calculation, the method is modified from Fredrickson's
            xi[i] = 0.5*(w[i]+w[i+sb->get_MM()]-pc->get_chi_n());
            // calculate output fields
            w_out[i]              = pc->get_chi_n()*phib[i] + xi[i];
            w_out[i+sb->get_MM()] = pc->get_chi_n()*phia[i] + xi[i];
        }
        sb->zero_mean(&w_out[0]);
        sb->zero_mean(&w_out[sb->get_MM()]);

        // error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level;
        for(int i=0; i<2*sb->get_MM(); i++)
            w_diff[i] = w_out[i]- w[i];
        error_level = sqrt(sb->multi_inner_product(2,w_diff,w_diff)/
                           (sb->multi_inner_product(2,w,w)+1.0));

        // print iteration # and error levels and check the mass conservation
        sum = (sb->integral(phia) + sb->integral(phib))/sb->get_volume() - 1.0;
        std::cout<< std::setw(8) << iter;
        std::cout<< std::setw(13) << std::setprecision(3) << std::scientific << sum ;
        std::cout<< std::setw(17) << std::setprecision(7) << std::scientific << QQ;
        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << energy_total;
        std::cout<< std::setw(15) << std::setprecision(9) << std::fixed << error_level << std::endl;

        // conditions to end the iteration
        if(error_level < tolerance) break;
        // calculte new fields using simple and Anderson mixing
        am->caculate_new_fields(w, w_out, w_diff, old_error_level, error_level);
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
    delete[] phia;
    delete[] phib;
    delete[] phitot;
    delete[] w_plus;
    delete[] w_minus;
    delete[] q1_init;
    delete[] q2_init;

    delete pc;
    delete sb;
    delete pseudo;
    delete am;
    delete factory;

    if (std::isnan(error_level) || std::abs(error_level-0.016984456) > 1e-7)
        return -1;

    return 0;
}