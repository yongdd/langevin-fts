#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <string>
#include <array>
#include <chrono>
#include <random>
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
        // Math constants
        const double PI = std::numbers::pi;

        // Input fields
        double *w, *w_out, *w_diff;
        double *xi, *w_plus, *w_minus;
        double *phi_a, *phi_b;

        // String to output file and print stream
        std::streamsize default_precision = std::cout.precision();

        // -------------- Initialize ------------
        int max_scft_iter = 500;
        double tolerance = 1e-9;

        double f = 0.5;
        double chi_n = 20.0;
        std::vector<int> nx = {64};
        std::vector<double> lx = {3.3};
        double ds = 1.0/100.0;

        int am_n_var = 2*nx[0];
        int am_max_hist = 20;
        double am_start_error = 1e-1;
        double am_mix_min = 0.1;
        double am_mix_init = 0.1;

        std::map<std::string, double> bond_lengths = {{"A",1.0}, {"B",1.0}};

        // Simple diblock
        std::vector<BlockInput> blocks =
        {
            {"A", f, 0, 1},
            {"B", 1.0-f, 1, 2},
        };

        const int M = nx[0];
        w       = new double[2*M];
        w_out   = new double[2*M];
        w_diff  = new double[2*M];
        xi      = new double[M];
        phi_a   = new double[M];
        phi_b   = new double[M];
        w_plus  = new double[M];
        w_minus = new double[M];

        // Choose platform
        std::vector<std::string> avail_platforms = PlatformSelector::avail_platforms();
        std::vector<std::string> chain_models = {"Continuous", "Discrete"};
        std::vector<bool> aggregate_propagator_computations = {false, true};

        for(std::string platform : avail_platforms)
        {
            if(platform != "cpu-mkl")
                continue;

            for(std::string chain_model : chain_models)
            {
                std::cout << "Testing: " << platform << ", " << chain_model << std::endl;

                for(bool aggregate : aggregate_propagator_computations)
                {
                    // Re-initialize fields with lamellar pattern
                    for(int i=0; i<nx[0]; i++)
                    {
                        double xx = (i+1)*2*PI/nx[0];
                        w[i] = std::cos(xx);
                        w[i+M] = -std::cos(xx);
                    }

                    bool reduce_memory_usage = false;
                    AbstractFactory<double> *factory = PlatformSelector::create_factory_real(platform, reduce_memory_usage);

                    // Create instances
                    ComputationBox<double>* cb = factory->create_computation_box(nx, lx, {});
                    Molecules* molecules = factory->create_molecules_information(chain_model, ds, bond_lengths);
                    molecules->add_polymer(1.0, blocks, {});
                    PropagatorComputationOptimizer* propagator_computation_optimizer =
                        new PropagatorComputationOptimizer(molecules, aggregate);
                    PropagatorComputation<double>* solver =
                        factory->create_pseudospectral_solver(cb, molecules, propagator_computation_optimizer);
                    AndersonMixing<double> *am = factory->create_anderson_mixing(am_n_var,
                        am_max_hist, am_start_error, am_mix_min, am_mix_init);

                    std::cout << "  Aggregation=" << std::boolalpha << aggregate << std::endl;

                    // -------------- SCFT iteration to find saddle point ------------
                    double error_level = 1.0e20;
                    double old_error_level = 1.0e20;

                    // Keep the level of field value
                    cb->zero_mean(&w[0]);
                    cb->zero_mean(&w[M]);

                    for(int iter=0; iter<max_scft_iter; iter++)
                    {
                        solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                        solver->compute_concentrations();
                        solver->get_total_concentration("A", phi_a);
                        solver->get_total_concentration("B", phi_b);

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

                    std::cout << "  SCFT error=" << std::scientific << std::setprecision(2) << error_level << std::endl;

                    // -------------- Numerical derivative test ------------
                    double dL = 0.0000001;
                    double old_lx0 = lx[0];

                    // Lambda to compute total energy
                    auto compute_energy = [&]() -> double {
                        for(int i=0; i<M; i++)
                        {
                            w_minus[i] = (w[i]-w[i+M])/2;
                            w_plus[i]  = (w[i]+w[i+M])/2;
                        }
                        double energy = cb->inner_product(w_minus,w_minus)/chi_n/cb->get_volume();
                        energy -= cb->integral(w_plus)/cb->get_volume();
                        for(int p=0; p<molecules->get_n_polymer_types(); p++){
                            Polymer& pc = molecules->get_polymer(p);
                            energy -= pc.get_volume_fraction()/pc.get_alpha()*log(solver->get_total_partition(p));
                        }
                        return energy;
                    };

                    // Compute numerical derivative dH/dL
                    lx[0] = old_lx0 + dL/2;
                    cb->set_lattice_parameters(lx);
                    solver->update_laplacian_operator();
                    solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                    double energy_total_1 = compute_energy();

                    lx[0] = old_lx0 - dL/2;
                    cb->set_lattice_parameters(lx);
                    solver->update_laplacian_operator();
                    solver->compute_propagators({{"A",&w[0]},{"B",&w[M]}},{});
                    double energy_total_2 = compute_energy();

                    // Reset to original
                    lx[0] = old_lx0;
                    cb->set_lattice_parameters(lx);
                    solver->update_laplacian_operator();

                    double dh_dl = (energy_total_1 - energy_total_2) / dL;
                    solver->compute_stress();
                    auto stress = solver->get_stress();

                    std::cout << std::setprecision(6);
                    std::cout << "  dH/dL : " << dh_dl << std::endl;
                    std::cout << "  Stress : " << stress[0] << std::endl;

                    double relative_stress_error = std::abs(dh_dl - stress[0]) / std::abs(stress[0]);
                    std::cout << "  Relative stress error : " << relative_stress_error << std::endl;

                    if (!std::isfinite(relative_stress_error) || relative_stress_error > 1e-3) {
                        std::cout << "ERROR: Numerical derivative does not match stress!" << std::endl;
                        return -1;
                    }

                    delete molecules;
                    delete propagator_computation_optimizer;
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
