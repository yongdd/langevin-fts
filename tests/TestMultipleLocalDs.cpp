/**
 * @file TestMultipleLocalDs.cpp
 * @brief Test propagator computation with multiple local_ds values.
 *
 * This test verifies that the propagator computation works correctly
 * when blocks have different local_ds values due to non-integer
 * contour_length/ds ratios.
 *
 * Example with global_ds = 0.1:
 * - Block A: contour_length = 0.3, n_segment = 3, local_ds = 0.1
 * - Block B: contour_length = 0.35, n_segment = 3, local_ds = 0.1167
 *            (0.35/0.1 = 3.5, but rounds to 3 due to floating-point precision)
 * - Block A: contour_length = 0.5, n_segment = 5, local_ds = 0.1
 *
 * This results in two unique local_ds values: 0.1 and 0.1167
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>

#include "Exception.h"
#include "PropagatorComputationOptimizer.h"
#include "Molecules.h"
#include "Polymer.h"
#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#endif
#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaComputationContinuous.h"
#endif

int main()
{
    try
    {
        const int II{8};
        const int JJ{8};
        const int KK{8};
        const int M{II*JJ*KK};

        double Lx = 4.0;
        double Ly = 4.0;
        double Lz = 4.0;

        // Initialize random potential fields
        double w_a[M], w_b[M];
        for(int i=0; i<M; i++)
        {
            w_a[i] = 0.1 * ((double)rand() / RAND_MAX);
            w_b[i] = 0.1 * ((double)rand() / RAND_MAX);
        }

        //-------------- initialize ------------
        std::cout << "=== Test: Multiple Local Ds Values ===" << std::endl;
        std::cout << std::endl;

        // Use global_ds = 0.1
        // Blocks with different contour lengths that result in different local_ds:
        // - 0.3: n_segment = round(0.3/0.1) = 3, local_ds = 0.3/3 = 0.1
        // - 0.35: n_segment = 3 (due to float precision), local_ds = 0.35/3 = 0.1167
        // - 0.5: n_segment = round(0.5/0.1) = 5, local_ds = 0.5/5 = 0.1
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks =
        {
            {"A", 0.3,  0, 1},  // local_ds = 0.1
            {"B", 0.35, 1, 2},  // local_ds = 0.1167 (different!)
            {"A", 0.5,  2, 3},  // local_ds = 0.1
        };

        Molecules* molecules = new Molecules("Continuous", 0.1, bond_lengths);
        molecules->add_polymer(1.0, blocks, {});

        // Display polymer architecture
        molecules->display_architectures();
        std::cout << std::endl;

        PropagatorComputationOptimizer* propagator_computation_optimizer =
            new PropagatorComputationOptimizer(molecules, false);

        // Display mapping, blocks, and propagators
        propagator_computation_optimizer->display_blocks();
        propagator_computation_optimizer->display_propagators();

        // Verify we have two unique local_ds values
        const auto& mapping = molecules->get_contour_length_mapping();
        int n_unique_ds = mapping.get_n_unique_ds();
        std::cout << std::endl;
        std::cout << "Number of unique local_ds values: " << n_unique_ds << std::endl;

        if (n_unique_ds != 2)
        {
            std::cout << "FAILED: Expected 2 unique local_ds values, got " << n_unique_ds << std::endl;
            return -1;
        }
        std::cout << "OK: Found 2 unique local_ds values as expected" << std::endl;

        // Verify ds_index values
        int ds_idx_03 = mapping.get_ds_index(0.3);
        int ds_idx_035 = mapping.get_ds_index(0.35);
        int ds_idx_05 = mapping.get_ds_index(0.5);

        std::cout << std::endl;
        std::cout << "ds_index for contour_length=0.3:  " << ds_idx_03 << std::endl;
        std::cout << "ds_index for contour_length=0.35: " << ds_idx_035 << std::endl;
        std::cout << "ds_index for contour_length=0.5:  " << ds_idx_05 << std::endl;

        // 0.3 and 0.5 should have the same ds_index, 0.35 should be different
        if (ds_idx_03 != ds_idx_05)
        {
            std::cout << "FAILED: ds_index(0.3) and ds_index(0.5) should be equal" << std::endl;
            return -1;
        }
        if (ds_idx_035 == ds_idx_03)
        {
            std::cout << "FAILED: ds_index(0.35) should differ from ds_index(0.3)" << std::endl;
            return -1;
        }
        std::cout << "OK: ds_index values are correct" << std::endl;

        // Now test actual propagator computation
        std::cout << std::endl;
        std::cout << "Testing propagator computation..." << std::endl;

        std::vector<PropagatorComputation<double>*> solver_list;
        std::vector<ComputationBox<double>*> cb_list;
        std::vector<std::string> solver_name_list;

        #ifdef USE_CPU_MKL
        solver_name_list.push_back("cpu-mkl");
        cb_list.push_back(new CpuComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CpuComputationContinuous<double>(
            cb_list.back(), molecules, propagator_computation_optimizer, "pseudospectral"));
        #endif

        #ifdef USE_CUDA
        solver_name_list.push_back("cuda");
        cb_list.push_back(new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list.push_back(new CudaComputationContinuous<double>(
            cb_list.back(), molecules, propagator_computation_optimizer, "pseudospectral"));
        #endif

        // For each platform
        for(size_t n=0; n<solver_list.size(); n++)
        {
            PropagatorComputation<double>* solver = solver_list[n];

            std::cout << std::endl << "Running: " << solver_name_list[n] << std::endl;

            // Compute propagators
            solver->compute_propagators({{"A", w_a}, {"B", w_b}}, {});
            solver->compute_concentrations();

            // Get partition function
            const int p = 0;
            double Q = solver->get_total_partition(p);
            std::cout << "Partition function Q = " << Q << std::endl;

            // Get concentrations
            double phi_a[M] = {0.0};
            double phi_b[M] = {0.0};
            solver->get_total_concentration("A", phi_a);
            solver->get_total_concentration("B", phi_b);

            // Check that concentrations are reasonable (positive and sum ~ 1)
            double sum_phi = 0.0;
            bool all_positive = true;
            for(int i=0; i<M; i++)
            {
                sum_phi += (phi_a[i] + phi_b[i]);
                if (phi_a[i] < 0 || phi_b[i] < 0)
                    all_positive = false;
            }
            sum_phi /= M;

            std::cout << "Average total concentration: " << sum_phi << std::endl;

            if (!all_positive)
            {
                std::cout << "FAILED: Negative concentration detected" << std::endl;
                return -1;
            }
            // With random fields, concentration may deviate from 1.0
            // Use 10% tolerance since we're mainly testing that computation works
            if (std::abs(sum_phi - 1.0) > 0.1)
            {
                std::cout << "FAILED: Average concentration should be ~1.0, got " << sum_phi << std::endl;
                return -1;
            }
            std::cout << "OK: Propagator computation successful for " << solver_name_list[n] << std::endl;
        }

        // Cleanup
        for(size_t n=0; n<solver_list.size(); n++)
        {
            delete solver_list[n];
            delete cb_list[n];
        }
        delete propagator_computation_optimizer;
        delete molecules;

        std::cout << std::endl;
        std::cout << "=== All tests PASSED ===" << std::endl;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
