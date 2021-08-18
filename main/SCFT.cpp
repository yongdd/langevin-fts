
#include <iomanip>
#include <cmath>
#include <string>
#include <chrono>
#include <sstream>
#include <fstream>

#include "ParamParser.h"
#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "KernelFactory.h"

int main(int argc, char **argv)
{
    // math constatns
    const double PI = 3.14159265358979323846;
    // chrono timer
    std::chrono::system_clock::time_point chrono_start, chrono_end;
    std::chrono::duration<double> time_duration;
    // iter = number of iteration steps, maxiter = maximum number of iteration steps
    int iter, maxiter;
    // QQ = total partition function
    double QQ, energy_chain, energy_field, energy_total, energy_old;
    // error_level = variable to check convergence of the iteration
    double error_level, old_error_level, tolerance;
    // input and output fields, xi is temporary storage for pressures
    double *w, *w_out, *w_diff;  // n_comp * MM
    double *xi, *w_plus, *w_minus; // MM
    // initial value of q, q_dagger
    double *q1_init, *q2_init;
    // segment concentration
    double *phia, *phib, *phitot;
    // input parameters
    int nx[3], NN;
    double lx[3], f, chi_n;
    // platform type, (CUDA, CPU_MKL or CPU_FFTW)
    std::string str_platform;
    // Anderson mixing parmeters
    int max_anderson;
    double start_anderson_error;
    double mix_min, mix_init;
    // string to output file and print stream
    std::ofstream print_stream;
    std::stringstream ss;
    std::string print_file_name;
    // temp
    int idx;
    double sum;

    // -------------- initialize ------------
    // read file name
    if(argc < 2)
    {
        std::cout<< "Input parameter file is required, e.g, 'scft.out inputs' " << std::endl;
        exit(-1);
    }
    // initialize ParamParser
    ParamParser& pp = ParamParser::get_instance();
    pp.read_param_file(argv[1], true);

    // choose platform
    KernelFactory *factory;
    if(!pp.get("platform", str_platform))
        factory = new KernelFactory();
    else
        factory = new KernelFactory(str_platform);

    // read simulation box parameters
    if(!pp.get("geometry.grids", nx, 3))
    {
        std::cout<< "geometry.grids is not specified." << std::endl;
        exit(-1);
    }
    if(!pp.get("geometry.box_size", lx, 3))
    {
        std::cout<< "geometry.box_size is not specified." << std::endl;
        exit(-1);
    }
    // read chain parameters
    if(!pp.get("chain.a_fraction", f))
    {
        std::cout<< "chain.a_fraction is not specified." << std::endl;
        exit(-1);
    }
    if(!pp.get("chain.contour_step", NN))
    {
        std::cout<< "chain.contour_step is not specified." << std::endl;
        exit(-1);
    }
    if(!pp.get("chain.chi_n", chi_n))
    {
        std::cout<< "chain.chi_n is not specified." << std::endl;
        exit(-1);
    }
    // read Anderson mixing parameters
    // anderson mixing begin if error level becomes less then start_anderson_error
    if(!pp.get("am.start_error", start_anderson_error)) start_anderson_error = 0.01;
    // max number of previous steps to calculate new field
    if(!pp.get("am.step_max", max_anderson)) max_anderson = 10;
    // minimum mixing parameter
    if(!pp.get("am.mix_min", mix_min)) mix_min = 0.01;
    // initial mixing parameter
    if(!pp.get("am.mix_init", mix_init)) mix_init  = 0.1;

    // read iteration parameters
    if(!pp.get("iter.tolerance", tolerance)) tolerance = 5.0e-9;
    if(!pp.get("iter.step_saddle", maxiter)) maxiter   = 10;

    // create instances and assign to the variables of base classs
    // for the dynamic binding
    PolymerChain *pc = factory->create_polymer_chain(f, NN, chi_n);
    SimulationBox *sb = factory->create_simulation_box(nx, lx);
    Pseudo *pseudo = factory->create_pseudo(sb, pc);
    AndersonMixing *am = factory->create_anderson_mixing(
                             sb, 2, max_anderson, start_anderson_error, mix_min, mix_init);
    // assign large initial value for the energy and error
    energy_total = 1.0e20;
    error_level = 1.0e20;

    // -------------- print simulation parameters ------------
    //ss << std::setw(4) << std::setfill('0') << std::lround(chi_n*100);
    //print_file_name = "print_" + ss.str() + ".txt" ;
    //print_stream->open(print_file_name, std::ios_base::app);
    std::cout<< "---------- Simulation Parameters ----------" << std::endl;

    std::cout << "Box Dimension: 3" << std::endl;
    std::cout << "Precision: 8" << std::endl;
    std::cout << "chi_n, f, NN: " << pc->chi_n << " " << pc->f << " " << pc->NN << std::endl;
    std::cout << "Nx: " << sb->nx[0] << " " << sb->nx[1] << " " << sb->nx[2] << std::endl;
    std::cout << "Lx: " << sb->lx[0] << " " << sb->lx[1] << " " << sb->lx[2] << std::endl;
    std::cout << "dx: " << sb->dx[0] << " " << sb->dx[1] << " " << sb->dx[2] << std::endl;
    sum = 0.0;
    for(int i=0; i<sb->MM; i++)
        sum += sb->dv[i];
    std::cout << "volume, sum(dv):  " << sb->volume << " " << sum << std::endl;

    //-------------- allocate array ------------
    w       = new double[sb->MM*2];
    w_out   = new double[sb->MM*2];
    w_diff  = new double[sb->MM*2];
    xi      = new double[sb->MM];
    phia    = new double[sb->MM];
    phib    = new double[sb->MM];
    phitot  = new double[sb->MM];
    w_plus  = new double[sb->MM];
    w_minus = new double[sb->MM];
    q1_init = new double[sb->MM];
    q2_init = new double[sb->MM];

    //-------------- setup fields ------------
    //call random_number(phia)
    //   phia = reshape( phia, (/ x_hi-x_lo+1,y_hi-y_lo+1,z_hi-z_lo+1 /), order = (/ 3, 2, 1 /))
    //   call random_number(phia(:,:,z_lo))
    //   do k=z_lo,z_hi
    //     phia(:,:,k) = phia(:,:,z_lo)
    //   end do

    std::cout<< "wminus and wplus are initialized to a given test fields." << std::endl;
    for(int i=0; i<sb->nx[0]; i++)
        for(int j=0; j<sb->nx[1]; j++)
            for(int k=0; k<sb->nx[2]; k++)
            {
                idx = i*sb->nx[1]*sb->nx[2] + j*sb->nx[2] + k;
                phia[idx]= cos(2.0*PI*i/4.68)*cos(2.0*PI*j/3.48)*cos(2.0*PI*k/2.74)*0.1;
            }

    for(int i=0; i<sb->MM; i++)
    {
        phib[i] = 1.0 - phia[i];
        w[i]          = chi_n*phib[i];
        w[i + sb->MM] = chi_n*phia[i];
    }

    // keep the level of field value
    sb->zero_mean(&w[0]);
    sb->zero_mean(&w[sb->MM]);

    // free end initial condition. q1 is q and q2 is qdagger.
    // q1 starts from A end and q2 starts from B end.
    for(int i=0; i<sb->MM; i++)
    {
        q1_init[i] = 1.0;
        q2_init[i] = 1.0;
    }
    //print_stream->close();

    //------------------ run ----------------------
    std::cout<< "---------- Run ----------" << std::endl;
    std::cout<< "iteration, mass error, total_partition, energy_total, error_level" << std::endl;
    chrono_start = std::chrono::system_clock::now();
    // iteration begins here
    for(int iter=0; iter<maxiter; iter++)
    {
        // for the given fields find the polymer statistics
        pseudo->find_phi(phia, phib,q1_init,q2_init,&w[0],&w[sb->MM],QQ);

        // calculate the total energy
        energy_old = energy_total;
        for(int i=0; i<sb->MM; i++)
        {
            w_minus[i] = (w[i]-w[i + sb->MM])/2;
            w_plus[i]  = (w[i]+w[i + sb->MM])/2;
        }
        energy_total  = -log(QQ/sb->volume);
        energy_total += sb->inner_product(w_minus,w_minus)/chi_n/sb->volume;
        energy_total += sb->integral(w_plus)/sb->volume;
        //energy_total += sb->inner_product(ext_w_minus,ext_w_minus)/chi_b/sb->volume;

        for(int i=0; i<sb->MM; i++)
        {
            // calculate pressure field for the new field calculation, the method is modified from Fredrickson's
            xi[i] = 0.5*(w[i]+w[i+sb->MM]-chi_n);
            // calculate output fields
            w_out[i]        = chi_n*phib[i] + xi[i];
            w_out[i+sb->MM] = chi_n*phia[i] + xi[i];
        }
        sb->zero_mean(&w_out[0]);
        sb->zero_mean(&w_out[sb->MM]);

        // error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level;
        for(int i=0; i<2*sb->MM; i++)
            w_diff[i] = w_out[i]- w[i];
        error_level = sqrt(sb->multi_inner_product(2,w_diff,w_diff)/
                           (sb->multi_inner_product(2,w,w)+1.0));

        // print iteration # and error levels and check the mass conservation
        sum = (sb->integral(phia) + sb->integral(phib))/sb->volume - 1.0;
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
    std::cout<< time_duration.count() << " " << time_duration.count()/maxiter << std::endl;
    //------------- write the final output -------------
    //write (filename, '( "fields_", I0.6, "_", I0.4, "_", I0.7, ".dat")' ) &
    //nint(chiW), nint(chiN*100), nint(Lx*1000)
    //call write_data(filename)

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

    pp.display_usage_info();

    /*
    !------------- write the final output -------------
    ! this subroutine write the final output
    subroutine write_data(filename)
    character (len=*), intent(in) :: filename
    integer :: i, j, k
    open(30,file=filename,status='unknown')
    write(30,'(A,I8)') "iter : ", iter
    write(30,'(A,F8.3)') "chain.chiN : ", chiN
    write(30,'(A,F8.3)') "chain.a_fraction : ", f
    write(30,'(A,F8.3)') "chain.surface_interaction : ", chiW
    write(30,'(A,I8)')   "chain.contour_step : ",  NN
    write(30,'(A,I8)')   "chain.contour_step_A : ", NNf
    write(30,'(A,3I8)')   "geometry.grids_lower_bound : ", x_lo, y_lo, z_lo
    write(30,'(A,3I8)')   "geometry.grids_upper_bound : ", x_hi, y_hi, z_hi
    write(30,'(A,3F8.3)') "geometry.box_size : ", Lx, Ly, Lz
    write(30,'(A,F13.9)') "energy_total : ", energy_total
    write(30,'(A,F13.9)') "error_level : ", error_level

    write(30,*) " "
    write(30,*) "DATA      # w_a, phi_a, w_b, phi_b"
    do i=x_lo,x_hi
    do j=y_lo,y_hi
    do k=z_lo,z_hi
      write(30,'(5F12.6)') w(i,j,k,1), phia(i,j,k), w(i,j,k,2), phib(i,j,k), phia(i,j,k)+phib(i,j,k)
    end do
    end do
    end do
    close(30)
    end subroutine
    end module scft
    */
}
