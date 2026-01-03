#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <string>
#include <array>
#include <chrono>

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
        // Math constants
        const double PI = std::numbers::pi;

        double energy_total;
        double error_level, old_error_level;
        double *w, *w_out, *w_diff;
        double *xi, *w_plus, *w_minus;
        double *phi_a, *phi_b, *phi_tot;

        std::streamsize default_precision = std::cout.precision();

        // -------------- Initialize ------------
        int max_scft_iter = 500;
        double tolerance = 1e-9;

        double f = 0.36;
        double chi_n = 20.0;
        std::vector<int> nx = {33, 29};
        std::vector<double> lx = {3.2, 3.4};
        double ds = 1.0/100;

        int am_n_var = 2*nx[0]*nx[1];
        int am_max_hist = 20;
        double am_start_error = 1e-1;
        double am_mix_min = 0.1;
        double am_mix_init = 0.1;

        const int M = nx[0]*nx[1];

        std::vector<BlockInput> blocks =
        {
            {"A",    f, 0, 1},
            {"B",1.0-f, 1, 2},
        };

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

        bool reduce_memory_usage = false;

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Discrete", "Continuous"};
        std::vector<bool> aggregate_propagator_computations = {false, true};

        for(std::string platform : avail_platforms)
        {
            for(std::string chain_model : chain_models)
            {
                std::cout << "Testing: " << platform << ", " << chain_model << std::endl;
                std::vector<double> model_stress_list_0;
                std::vector<double> model_stress_list_1;

                for(bool aggregate_propagator_computation : aggregate_propagator_computations)
                {
                    AbstractFactory<double> *factory = PlatformSelector::create_factory_real(platform, reduce_memory_usage);
                    factory->display_info();

                    // Create instances
                    ComputationBox<double>* cb = factory->create_computation_box(nx, lx, {});
                    Molecules* molecules = factory->create_molecules_information(chain_model, ds, {{"A",1.0}, {"B",1.0}});
                    molecules->add_polymer(1.0, blocks, {});
                    PropagatorComputationOptimizer* propagator_computation_optimizer =
                        new PropagatorComputationOptimizer(molecules, aggregate_propagator_computation);
                    PropagatorComputation<double>* solver =
                        factory->create_pseudospectral_solver(cb, molecules, propagator_computation_optimizer);
                    AndersonMixing<double> *am = factory->create_anderson_mixing(am_n_var,
                        am_max_hist, am_start_error, am_mix_min, am_mix_init);

                    std::cout << "Chain Model: " << molecules->get_model_name() << std::endl;
                    std::cout << "Using Aggregation: " << std::boolalpha << aggregate_propagator_computation << std::endl;

                    // Load pre-converged fields from file
                    std::string line;
                    std::ifstream input_field_file;
                    if(molecules->get_model_name() == "continuous")
                        input_field_file.open("Stress2D_ContinuousInput.txt");
                    else if(molecules->get_model_name() == "discrete")
                        input_field_file.open("Stress2D_DiscreteInput.txt");

                    if (input_field_file.is_open())
                    {
                        for(int i=0; i<2*M ; i++)
                        {
                            std::getline(input_field_file, line);
                            w[i] = std::stod(line);
                        }
                        input_field_file.close();
                    }
                    else
                    {
                        // Initialize to lamellar-like pattern if no input file
                        for(int i=0; i<nx[0]; i++)
                        {
                            double xx = (i+1)*2*PI/nx[0];
                            for(int j=0; j<nx[1]; j++)
                            {
                                int idx = i*nx[1] + j;
                                w[idx] = std::cos(xx);
                                w[idx+M] = -std::cos(xx);
                            }
                        }
                    }

                    // Keep the level of field value
                    cb->zero_mean(&w[0]);
                    cb->zero_mean(&w[M]);

                    // Assign large initial value for the energy and error
                    energy_total = 1.0e20;
                    error_level = 1.0e20;

                    // SCFT Iteration
                    for(int iter=0; iter<max_scft_iter; iter++)
                    {
                        solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                        solver->compute_concentrations();
                        solver->get_total_concentration("A", phi_a);
                        solver->get_total_concentration("B", phi_b);

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
                            xi[i] = 0.5*(w[i]+w[i+M]-chi_n);
                            w_out[i]   = chi_n*phi_b[i] + xi[i];
                            w_out[i+M] = chi_n*phi_a[i] + xi[i];
                        }
                        cb->zero_mean(&w_out[0]);
                        cb->zero_mean(&w_out[M]);

                        old_error_level = error_level;
                        for(int i=0; i<2*M; i++)
                            w_diff[i] = w_out[i]- w[i];
                        error_level = sqrt(cb->multi_inner_product(2,w_diff,w_diff)/
                                        (cb->multi_inner_product(2,w,w)+1.0));

                        if(error_level < tolerance) break;
                        am->calculate_new_fields(w, w, w_diff, old_error_level, error_level);
                    }

                    // Compute stress at saddle point
                    solver->compute_stress();
                    auto stress = solver->get_stress();

                    std::cout << "  Aggregation=" << std::boolalpha << aggregate_propagator_computation;
                    std::cout << ", SCFT error=" << std::scientific << std::setprecision(2) << error_level;
                    std::cout << ", Stress[0]=" << std::setprecision(6) << stress[0];
                    std::cout << ", Stress[1]=" << std::setprecision(6) << stress[1] << std::endl;

                    model_stress_list_0.push_back(stress[0]);
                    model_stress_list_1.push_back(stress[1]);

                    delete molecules;
                    delete propagator_computation_optimizer;
                    delete cb;
                    delete solver;
                    delete am;
                    delete factory;
                }

                // Check consistency between aggregated and non-aggregated for both stress components
                if (model_stress_list_0.size() >= 2) {
                    // Check stress[0] consistency
                    double diff0 = std::abs(model_stress_list_0[0] - model_stress_list_0[1]);
                    double max_val0 = std::max(std::abs(model_stress_list_0[0]), std::abs(model_stress_list_0[1]));
                    double rel_diff0 = diff0 / (max_val0 + 1e-10);
                    std::cout << "  Relative difference (stress[0], aggregation): " << rel_diff0 << std::endl;
                    if (rel_diff0 > 1e-7) {
                        std::cout << "ERROR: Stress[0] values differ too much between aggregated and non-aggregated!" << std::endl;
                        return -1;
                    }

                    // Check stress[1] consistency
                    double diff1 = std::abs(model_stress_list_1[0] - model_stress_list_1[1]);
                    double max_val1 = std::max(std::abs(model_stress_list_1[0]), std::abs(model_stress_list_1[1]));
                    double rel_diff1 = diff1 / (max_val1 + 1e-10);
                    std::cout << "  Relative difference (stress[1], aggregation): " << rel_diff1 << std::endl;
                    if (rel_diff1 > 1e-7) {
                        std::cout << "ERROR: Stress[1] values differ too much between aggregated and non-aggregated!" << std::endl;
                        return -1;
                    }
                }
            }
        }

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

        std::cout << "Test passed!" << std::endl;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
