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
        // Chrono timer
        std::chrono::system_clock::time_point chrono_start, chrono_end;

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

        // -------------- Initialize ------------
        // Platform type, [cuda, cpu-mkl]
        
        int max_scft_iter = 200;
        double tolerance = 1e-9;

        double chi_n = 10.0;
        std::vector<int> nx = {255};
        std::vector<double> lx = {1.5};
        double ds = 1.0/100;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.5}};
        std::vector<BlockInput> blocks_1 =
        {
            {"A", 0.6, 0, 1},
            {"A", 1.2, 0, 2},
            {"B", 1.2, 0, 5},
            {"B", 0.9, 0, 6},
            {"A", 0.9, 1, 4},
            {"A", 1.2, 1,15},

        };

        std::vector<BlockInput> blocks_2 =
        {
            {"A", 0.4, 0, 1},
            {"B", 0.6, 1, 2},
        };

        int am_n_var = 2*nx[0];  // A and B
        int am_max_hist = 20;
        double am_start_error = 8e-1;
        double am_mix_min = 0.1;
        double am_mix_init = 0.1;

        const int M = nx[0];
        //-------------- Allocate array ------------
        w       = new double[2*M];
        w_out   = new double[2*M];
        w_diff  = new double[2*M];
        xi      = new double[M];
        phi_a   = new double[M];
        phi_b   = new double[M];
        phi_tot = new double[M];
        w_plus  = new double[M];
        w_minus = new double[M];

        bool reduce_memory_usage=false;

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        std::vector<bool> aggregate_propagator_computations = {false, true};
        for(std::string platform : avail_platforms)
        {
            for(std::string chain_model : chain_models)
            {
                for(bool aggregate_propagator_computation : aggregate_propagator_computations)
                {
                    AbstractFactory *factory = PlatformSelector::create_factory(platform, reduce_memory_usage);
                    factory->display_info();

                    // Create instances and assign to the variables of base classes for the dynamic binding
                    ComputationBox *cb = factory->create_computation_box(nx, lx, {});
                    Molecules* molecules        = factory->create_molecules_information(chain_model, ds, bond_lengths);
                    molecules->add_polymer(0.7, blocks_1, {});
                    molecules->add_polymer(0.3, blocks_2, {});
                    PropagatorAnalyzer* propagator_analyzer= new PropagatorAnalyzer(molecules, aggregate_propagator_computation);
                    PropagatorComputation *solver     = factory->create_pseudospectral_solver(cb, molecules, propagator_analyzer);
                    AndersonMixing *am = factory->create_anderson_mixing(am_n_var,
                                        am_max_hist, am_start_error, am_mix_min, am_mix_init);

                    // -------------- Print simulation parameters ------------
                    std::cout << std::setprecision(default_precision);
                    std::cout << std::boolalpha;
                    // std::cout<< "---------- Simulation Parameters ----------" << std::endl;
                    // std::cout << "Box Dimension: " << cb->get_dim() << std::endl;
                    std::cout << "Chain Model: " << molecules->get_model_name() << std::endl;
                    std::cout << "Using Aggregation: " << aggregate_propagator_computation << std::endl;
                    // std::cout << "chi_n, f: " << chi_n << " " << f << " "  << std::endl;
                    // std::cout << "Nx: " << cb->get_nx(0) << " " << cb->get_nx(1) << " " << cb->get_nx(2) << std::endl;
                    // std::cout << "Lx: " << cb->get_lx(0) << " " << cb->get_lx(1) << " " << cb->get_lx(2) << std::endl;
                    // std::cout << "dx: " << cb->get_dx(0) << " " << cb->get_dx(1) << " " << cb->get_dx(2) << std::endl;

                    // // std::cout<< "w_a and w_b are initialized to a lamellar." << std::endl;
                    // for(int i=0; i<cb->get_nx(0); i++)
                    //     for(int j=0; j<cb->get_nx(1); j++)
                    //         for(int k=0; k<cb->get_nx(2); k++)
                    //         {
                    //             w[k]   =  cos(2.0*PI*k/cb->get_nx(2))*10;
                    //             w[k+M] = -cos(2.0*PI*k/cb->get_nx(2))*10;
                    //         }

                    std::string line;
                    std::ifstream input_field_file;
                    if(molecules->get_model_name() == "continuous")
                        input_field_file.open("Stress1D_ContinuousInput.txt");
                    else if(molecules->get_model_name() == "discrete")
                        input_field_file.open("Stress1D_DiscreteInput.txt");

                    if (input_field_file.is_open())
                    {
                        // std::cout << "input_field_file" << std::endl;
                        for(int i=0; i<2*M ; i++)
                        {
                            std::getline(input_field_file, line);
                            w[i] = std::stod(line);
                            // std::cout << line << " " << w[i] << std::endl;
                        }
                        input_field_file.close();
                    }
                    else
                    {
                        std::cout << "Could not open input file." << std::endl;
                    }

                    // Keep the level of field value
                    cb->zero_mean(&w[0]);
                    cb->zero_mean(&w[M]);

                    // Assign large initial value for the energy and error
                    energy_total = 1.0e20;
                    error_level = 1.0e20;

                    //------------------ Run ----------------------
                    // std::cout<< "---------- Run ----------" << std::endl;
                    // std::cout<< "iteration, mass error, total partitions, total energy, error level" << std::endl;
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
                            w_minus[i] = (w[i]-w[i+M])/2;
                            w_plus[i]  = (w[i]+w[i+M])/2;
                        }

                        energy_total = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                        energy_total -= cb->integral(w_plus)/cb->get_volume();
                        for(int p=0; p<molecules->get_n_polymer_types(); p++){
                            Polymer& pc = molecules->get_polymer(p);
                            energy_total -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
                        }

                        for(int i=0; i<M; i++)
                        {
                            // Calculate pressure field for the new field calculation
                            xi[i] = 0.5*(w[i]+w[i+M]-chi_n);
                            // Calculate output fields
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
                        std::cout << "error_level: " << error_level << std::endl;

                        // // print iteration # and error levels and check the mass conservation
                        // sum = (cb->integral(phi_a) + cb->integral(phi_b))/cb->get_volume() - 1.0;

                        // Conditions to end the iteration
                        if(error_level < tolerance) break;

                        // Calculate new fields using simple and Anderson mixing
                                        //w_new, w_current, w_diff
                        am->calculate_new_fields(w, w, w_diff, old_error_level, error_level);
                    }

                    // if(molecules->get_model_name() == "continuous")
                    // {
                    //     std::ofstream output_field_file("Stress1D_ContinuousInput.txt");
                    //     if (output_field_file.is_open())
                    //     {
                    //         for(int i=0; i<2*M ; i++){
                    //             output_field_file << std::setprecision(10) << w[i] << std::endl;
                    //         }
                    //         output_field_file.close();
                    //     }
                    // }
                    // else if(molecules->get_model_name() == "discrete")
                    // {
                    //     std::ofstream output_field_file("Stress1D_DiscreteInput.txt");
                    //     if (output_field_file.is_open())
                    //     {
                    //         for(int i=0; i<2*M ; i++){
                    //             output_field_file << std::setprecision(10) << w[i] << std::endl;
                    //         }
                    //         output_field_file.close();
                    //     }
                    // }

                    double dL = 0.0000001;
                    double old_lx = lx[0];
                    //----------- Compute derivate of H: lx + delta ----------------
                    lx[0] = old_lx + dL/2;
                    cb->set_lx(lx);
                    solver->update_laplacian_operator();

                    // For the given fields find the polymer statistics
                    solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                    solver->get_total_concentration("A", phi_a);
                    solver->get_total_concentration("B", phi_b);

                    // Calculate the total energy
                    for(int i=0; i<M; i++)
                    {
                        w_minus[i] = (w[i]-w[i+M])/2;
                        w_plus[i]  = (w[i]+w[i+M])/2;
                    }

                    double energy_total_1 = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                    energy_total_1 -= cb->integral(w_plus)/cb->get_volume();
                    for(int p=0; p<molecules->get_n_polymer_types(); p++){
                        Polymer& pc = molecules->get_polymer(p);
                        energy_total_1 -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
                    }

                    //----------- Compute derivate of H: lx - delta ----------------
                    lx[0] = old_lx - dL/2;
                    cb->set_lx(lx);
                    solver->update_laplacian_operator();

                    // For the given fields find the polymer statistics
                    solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                    solver->get_total_concentration("A", phi_a);
                    solver->get_total_concentration("B", phi_b);

                    // Calculate the total energy
                    for(int i=0; i<M; i++)
                    {
                        w_minus[i] = (w[i]-w[i+M])/2;
                        w_plus[i]  = (w[i]+w[i+M])/2;
                    }

                    double energy_total_2 = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                    energy_total_2 -= cb->integral(w_plus)/cb->get_volume();
                    for(int p=0; p<molecules->get_n_polymer_types(); p++){
                        Polymer& pc = molecules->get_polymer(p);
                        energy_total_2 -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
                    }

                    // Compute stress
                    double dh_dl = (energy_total_1-energy_total_2)/dL;
                    solver->compute_stress();
                    auto stress = solver->get_stress();

                    std:: cout << "dH/dL : " << dh_dl << std::endl;
                    std:: cout << "Stress : " << stress[0] << std::endl;
                    double relative_stress_error = std::abs(dh_dl-stress[0])/std::abs(stress[0]);
                    std:: cout << "Relative stress error : " << relative_stress_error << std::endl;
                    if (!std::isfinite(relative_stress_error) || std::abs(relative_stress_error) > 1e-3)
                        return -1;

                    //------------- Finalize -------------
                    delete molecules;
                    delete propagator_analyzer;
                    delete cb;
                    delete solver;
                    delete am;
                    delete factory;
                }
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
