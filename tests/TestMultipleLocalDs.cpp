/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

/**
 * @file TestMultipleLocalDs.cpp
 * @brief Test propagator computation with multiple local_ds values.
 *
 * This test verifies that the propagator computation works correctly
 * when blocks have different local_ds values due to non-integer
 * contour_length/ds ratios.
 *
 * **Test 1: Linear triblock**
 * - Block A: contour_length = 0.3, n_segment = 3, local_ds = 0.1
 * - Block B: contour_length = 0.35, n_segment = 4, local_ds = 0.0875
 * - Block A: contour_length = 0.5, n_segment = 5, local_ds = 0.1
 *
 * **Test 2: 6-arm star polymer**
 * - 3 arms (A-type): contour_length = 0.3, n_segment = 3, local_ds = 0.1
 * - 3 arms (B-type): contour_length = 0.35, n_segment = 4, local_ds = 0.0875
 *
 * **Test 3: 4 homopolymers with shared local_ds**
 * - Polymer 1: contour_length = 0.3,   n_segment = 3, local_ds = 0.1
 * - Polymer 2: contour_length = 0.5,   n_segment = 5, local_ds = 0.1
 * - Polymer 3: contour_length = 0.175, n_segment = 2, local_ds = 0.0875
 * - Polymer 4: contour_length = 0.35,  n_segment = 4, local_ds = 0.0875
 * (4 different lengths, but pairs share local_ds values)
 *
 * **Test 4: Bottle-brush polymer (backbone + 4 side chains)**
 * - Backbone: 5 segments of B-type, contour_length = 0.2, local_ds = 0.1
 * - Side chains: same lengths as Test 3 homopolymers attached to backbone
 * (5 different lengths, but only 2 unique local_ds values)
 *
 * All tests use global_ds = 0.1 and result in two unique local_ds: 0.1 and 0.0875
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
#ifdef USE_CPU_FFTW
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#include "CpuComputationReduceMemoryContinuous.h"
#endif
#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaComputationContinuous.h"
#include "CudaComputationReduceMemoryContinuous.h"
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
        // - 0.35: n_segment = round(0.35/0.1) = 4, local_ds = 0.35/4 = 0.0875
        // - 0.5: n_segment = round(0.5/0.1) = 5, local_ds = 0.5/5 = 0.1
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks =
        {
            {"A", 0.3,  0, 1},  // local_ds = 0.1
            {"B", 0.35, 1, 2},  // local_ds = 0.0875 (different!)
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

        #ifdef USE_CPU_FFTW
        solver_name_list.push_back("cpu-fftw");
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

        //-------------- Test 2: 6-arm star polymer ------------
        std::cout << std::endl;
        std::cout << "=== Test 2: 6-Arm Star Polymer with Two Local Ds Values ===" << std::endl;
        std::cout << std::endl;

        // 6-arm star polymer where:
        // - 3 arms have contour_length = 0.3, n_segment = 3, local_ds = 0.1
        // - 3 arms have contour_length = 0.35, n_segment = 4, local_ds = 0.0875
        // All arms connect to central junction (node 0)
        //
        //       (1)  (2)  (3)         Arms 1,2,3: A-type (0.3)
        //         \   |   /
        //          \  |  /
        //            (0)
        //          /  |  \
        //         /   |   \
        //       (4)  (5)  (6)         Arms 4,5,6: B-type (0.35)
        //

        std::map<std::string, double> bond_lengths_star = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks_star =
        {
            {"A", 0.3,  0, 1},   // Arm 1: local_ds = 0.1
            {"A", 0.3,  0, 2},   // Arm 2: local_ds = 0.1
            {"A", 0.3,  0, 3},   // Arm 3: local_ds = 0.1
            {"B", 0.35, 0, 4},   // Arm 4: local_ds = 0.0875
            {"B", 0.35, 0, 5},   // Arm 5: local_ds = 0.0875
            {"B", 0.35, 0, 6},   // Arm 6: local_ds = 0.0875
        };

        Molecules* molecules_star = new Molecules("Continuous", 0.1, bond_lengths_star);
        molecules_star->add_polymer(1.0, blocks_star, {});

        // Display polymer architecture
        molecules_star->display_architectures();
        std::cout << std::endl;

        PropagatorComputationOptimizer* optimizer_star =
            new PropagatorComputationOptimizer(molecules_star, false);

        // Display mapping, blocks, and propagators
        optimizer_star->display_blocks();
        optimizer_star->display_propagators();

        // Verify local_ds values
        const auto& mapping_star = molecules_star->get_contour_length_mapping();
        int n_unique_ds_star = mapping_star.get_n_unique_ds();
        std::cout << std::endl;
        std::cout << "Number of unique local_ds values: " << n_unique_ds_star << std::endl;

        if (n_unique_ds_star != 2)
        {
            std::cout << "FAILED: Expected 2 unique local_ds values, got " << n_unique_ds_star << std::endl;
            return -1;
        }
        std::cout << "OK: Found 2 unique local_ds values as expected" << std::endl;

        // Verify n_segment values
        int n_seg_03 = mapping_star.get_n_segment(0.3);
        int n_seg_035 = mapping_star.get_n_segment(0.35);
        std::cout << std::endl;
        std::cout << "n_segment for contour_length=0.3:  " << n_seg_03 << " (expected 3)" << std::endl;
        std::cout << "n_segment for contour_length=0.35: " << n_seg_035 << " (expected 4)" << std::endl;

        if (n_seg_03 != 3)
        {
            std::cout << "FAILED: n_segment(0.3) should be 3" << std::endl;
            return -1;
        }
        if (n_seg_035 != 4)
        {
            std::cout << "FAILED: n_segment(0.35) should be 4 (round(3.5) = 4)" << std::endl;
            return -1;
        }

        // Verify ds_index: same length_index should give same ds_index
        int ds_idx_03_star = mapping_star.get_ds_index(0.3);
        int ds_idx_035_star = mapping_star.get_ds_index(0.35);
        std::cout << std::endl;
        std::cout << "ds_index for contour_length=0.3:  " << ds_idx_03_star << std::endl;
        std::cout << "ds_index for contour_length=0.35: " << ds_idx_035_star << std::endl;

        if (ds_idx_03_star == ds_idx_035_star)
        {
            std::cout << "FAILED: ds_index(0.3) and ds_index(0.35) should differ" << std::endl;
            return -1;
        }
        std::cout << "OK: ds_index values are different as expected" << std::endl;

        // Test actual propagator computation for star polymer
        std::cout << std::endl;
        std::cout << "Testing propagator computation for 6-arm star..." << std::endl;

        std::vector<PropagatorComputation<double>*> solver_list_star;
        std::vector<ComputationBox<double>*> cb_list_star;
        std::vector<std::string> solver_name_list_star;

        #ifdef USE_CPU_FFTW
        solver_name_list_star.push_back("cpu-fftw");
        cb_list_star.push_back(new CpuComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list_star.push_back(new CpuComputationContinuous<double>(
            cb_list_star.back(), molecules_star, optimizer_star, "pseudospectral"));
        #endif

        #ifdef USE_CUDA
        solver_name_list_star.push_back("cuda");
        cb_list_star.push_back(new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list_star.push_back(new CudaComputationContinuous<double>(
            cb_list_star.back(), molecules_star, optimizer_star, "pseudospectral"));
        #endif

        for(size_t n=0; n<solver_list_star.size(); n++)
        {
            PropagatorComputation<double>* solver = solver_list_star[n];

            std::cout << std::endl << "Running: " << solver_name_list_star[n] << std::endl;

            // Compute propagators
            solver->compute_propagators({{"A", w_a}, {"B", w_b}}, {});
            solver->compute_concentrations();

            // Get partition function
            double Q_star = solver->get_total_partition(0);
            std::cout << "Partition function Q = " << Q_star << std::endl;

            // Get concentrations
            double phi_a_star[M] = {0.0};
            double phi_b_star[M] = {0.0};
            solver->get_total_concentration("A", phi_a_star);
            solver->get_total_concentration("B", phi_b_star);

            // Check that concentrations are reasonable
            double sum_phi_star = 0.0;
            bool all_positive_star = true;
            for(int i=0; i<M; i++)
            {
                sum_phi_star += (phi_a_star[i] + phi_b_star[i]);
                if (phi_a_star[i] < 0 || phi_b_star[i] < 0)
                    all_positive_star = false;
            }
            sum_phi_star /= M;

            std::cout << "Average total concentration: " << sum_phi_star << std::endl;

            if (!all_positive_star)
            {
                std::cout << "FAILED: Negative concentration detected" << std::endl;
                return -1;
            }
            if (std::abs(sum_phi_star - 1.0) > 0.1)
            {
                std::cout << "FAILED: Average concentration should be ~1.0, got " << sum_phi_star << std::endl;
                return -1;
            }
            std::cout << "OK: Propagator computation successful for " << solver_name_list_star[n] << std::endl;
        }

        // Cleanup star polymer
        for(size_t n=0; n<solver_list_star.size(); n++)
        {
            delete solver_list_star[n];
            delete cb_list_star[n];
        }
        delete optimizer_star;
        delete molecules_star;

        //-------------- Test 3: 4 homopolymers with shared local_ds ------------
        std::cout << std::endl;
        std::cout << "=== Test 3: 4 Homopolymers with Shared Local Ds Values ===" << std::endl;
        std::cout << std::endl;

        // 4 homopolymers with different lengths but only 2 unique local_ds:
        // - Polymer 1: contour_length = 0.3,   n_segment = 3, local_ds = 0.1
        // - Polymer 2: contour_length = 0.5,   n_segment = 5, local_ds = 0.1
        // - Polymer 3: contour_length = 0.175, n_segment = 2, local_ds = 0.0875
        // - Polymer 4: contour_length = 0.35,  n_segment = 4, local_ds = 0.0875
        //
        // Polymers 1,2 share local_ds = 0.1
        // Polymers 3,4 share local_ds = 0.0875

        std::map<std::string, double> bond_lengths_homo = {{"A", 1.0}};

        Molecules* molecules_homo = new Molecules("Continuous", 0.1, bond_lengths_homo);
        molecules_homo->add_polymer(0.25, {{"A", 0.3,   0, 1}}, {});  // local_ds = 0.1
        molecules_homo->add_polymer(0.25, {{"A", 0.5,   0, 1}}, {});  // local_ds = 0.1
        molecules_homo->add_polymer(0.25, {{"A", 0.175, 0, 1}}, {});  // local_ds = 0.0875
        molecules_homo->add_polymer(0.25, {{"A", 0.35,  0, 1}}, {});  // local_ds = 0.0875

        // Display polymer architectures
        molecules_homo->display_architectures();
        std::cout << std::endl;

        PropagatorComputationOptimizer* optimizer_homo =
            new PropagatorComputationOptimizer(molecules_homo, false);

        optimizer_homo->display_blocks();
        optimizer_homo->display_propagators();

        // Verify mapping
        const auto& mapping_homo = molecules_homo->get_contour_length_mapping();
        std::cout << std::endl;

        // Should have 4 unique lengths
        int n_unique_lengths_homo = mapping_homo.get_n_unique_lengths();
        std::cout << "Number of unique contour lengths: " << n_unique_lengths_homo << std::endl;
        if (n_unique_lengths_homo != 4)
        {
            std::cout << "FAILED: Expected 4 unique lengths, got " << n_unique_lengths_homo << std::endl;
            return -1;
        }

        // Should have 2 unique local_ds values
        int n_unique_ds_homo = mapping_homo.get_n_unique_ds();
        std::cout << "Number of unique local_ds values: " << n_unique_ds_homo << std::endl;
        if (n_unique_ds_homo != 2)
        {
            std::cout << "FAILED: Expected 2 unique local_ds values, got " << n_unique_ds_homo << std::endl;
            return -1;
        }
        std::cout << "OK: Found 4 unique lengths and 2 unique local_ds values" << std::endl;

        // Verify n_segment values
        std::cout << std::endl;
        std::cout << "n_segment for 0.175: " << mapping_homo.get_n_segment(0.175) << " (expected 2)" << std::endl;
        std::cout << "n_segment for 0.3:   " << mapping_homo.get_n_segment(0.3) << " (expected 3)" << std::endl;
        std::cout << "n_segment for 0.35:  " << mapping_homo.get_n_segment(0.35) << " (expected 4)" << std::endl;
        std::cout << "n_segment for 0.5:   " << mapping_homo.get_n_segment(0.5) << " (expected 5)" << std::endl;

        if (mapping_homo.get_n_segment(0.175) != 2 ||
            mapping_homo.get_n_segment(0.3) != 3 ||
            mapping_homo.get_n_segment(0.35) != 4 ||
            mapping_homo.get_n_segment(0.5) != 5)
        {
            std::cout << "FAILED: n_segment values incorrect" << std::endl;
            return -1;
        }

        // Verify ds_index sharing
        int ds_idx_homo_0175 = mapping_homo.get_ds_index(0.175);
        int ds_idx_homo_030 = mapping_homo.get_ds_index(0.3);
        int ds_idx_homo_035 = mapping_homo.get_ds_index(0.35);
        int ds_idx_homo_050 = mapping_homo.get_ds_index(0.5);

        std::cout << std::endl;
        std::cout << "ds_index for 0.175: " << ds_idx_homo_0175 << std::endl;
        std::cout << "ds_index for 0.3:   " << ds_idx_homo_030 << std::endl;
        std::cout << "ds_index for 0.35:  " << ds_idx_homo_035 << std::endl;
        std::cout << "ds_index for 0.5:   " << ds_idx_homo_050 << std::endl;

        // 0.175 and 0.35 should share ds_index (local_ds = 0.0875)
        // 0.3 and 0.5 should share ds_index (local_ds = 0.1)
        if (ds_idx_homo_0175 != ds_idx_homo_035)
        {
            std::cout << "FAILED: ds_index(0.175) and ds_index(0.35) should be equal" << std::endl;
            return -1;
        }
        if (ds_idx_homo_030 != ds_idx_homo_050)
        {
            std::cout << "FAILED: ds_index(0.3) and ds_index(0.5) should be equal" << std::endl;
            return -1;
        }
        if (ds_idx_homo_0175 == ds_idx_homo_030)
        {
            std::cout << "FAILED: ds_index(0.175) and ds_index(0.3) should differ" << std::endl;
            return -1;
        }
        std::cout << "OK: ds_index values correctly shared" << std::endl;

        // Test propagator computation
        std::cout << std::endl;
        std::cout << "Testing propagator computation for 4 homopolymers..." << std::endl;

        std::vector<PropagatorComputation<double>*> solver_list_homo;
        std::vector<ComputationBox<double>*> cb_list_homo;
        std::vector<std::string> solver_name_list_homo;

        #ifdef USE_CPU_FFTW
        solver_name_list_homo.push_back("cpu-fftw");
        cb_list_homo.push_back(new CpuComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list_homo.push_back(new CpuComputationContinuous<double>(
            cb_list_homo.back(), molecules_homo, optimizer_homo, "pseudospectral"));
        #endif

        #ifdef USE_CUDA
        solver_name_list_homo.push_back("cuda");
        cb_list_homo.push_back(new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list_homo.push_back(new CudaComputationContinuous<double>(
            cb_list_homo.back(), molecules_homo, optimizer_homo, "pseudospectral"));
        #endif

        for(size_t n=0; n<solver_list_homo.size(); n++)
        {
            PropagatorComputation<double>* solver = solver_list_homo[n];

            std::cout << std::endl << "Running: " << solver_name_list_homo[n] << std::endl;

            solver->compute_propagators({{"A", w_a}}, {});
            solver->compute_concentrations();

            // Get partition functions for all 4 polymers
            for(int p=0; p<4; p++)
            {
                double Q = solver->get_total_partition(p);
                std::cout << "Polymer " << p << " partition function Q = " << Q << std::endl;
            }

            // Get total concentration
            double phi_a_homo[M] = {0.0};
            solver->get_total_concentration("A", phi_a_homo);

            double sum_phi_homo = 0.0;
            bool all_positive_homo = true;
            for(int i=0; i<M; i++)
            {
                sum_phi_homo += phi_a_homo[i];
                if (phi_a_homo[i] < 0)
                    all_positive_homo = false;
            }
            sum_phi_homo /= M;

            std::cout << "Average total concentration: " << sum_phi_homo << std::endl;

            if (!all_positive_homo)
            {
                std::cout << "FAILED: Negative concentration detected" << std::endl;
                return -1;
            }
            if (std::abs(sum_phi_homo - 1.0) > 0.1)
            {
                std::cout << "FAILED: Average concentration should be ~1.0, got " << sum_phi_homo << std::endl;
                return -1;
            }
            std::cout << "OK: Propagator computation successful for " << solver_name_list_homo[n] << std::endl;
        }

        // Cleanup
        for(size_t n=0; n<solver_list_homo.size(); n++)
        {
            delete solver_list_homo[n];
            delete cb_list_homo[n];
        }
        delete optimizer_homo;
        delete molecules_homo;

        //-------------- Test 4: Bottle-brush with 4 side chains ------------
        std::cout << std::endl;
        std::cout << "=== Test 4: Bottle-Brush Polymer (Backbone + 4 Side Chains) ===" << std::endl;
        std::cout << std::endl;

        // Bottle-brush polymer with backbone (B-type) and 4 side chains (A-type)
        // Side chains have the same lengths as the 4 homopolymers in Test 3
        //
        //         (6)       (7)       (8)       (9)
        //          |         |         |         |
        //        A[0.3]    A[0.5]   A[0.175]  A[0.35]
        //          |         |         |         |
        //  (0)--B--(1)--B--(2)--B--(3)--B--(4)--B--(5)
        //      [0.2]    [0.2]    [0.2]    [0.2]    [0.2]
        //
        // Backbone: 5 segments of 0.2, n_segment=2, local_ds=0.1
        // Side chains: same as Test 3 homopolymers

        std::map<std::string, double> bond_lengths_brush = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks_brush =
        {
            {"B", 0.2,   0, 1},   // backbone segment, local_ds = 0.1
            {"A", 0.3,   1, 6},   // side chain 1, local_ds = 0.1
            {"B", 0.2,   1, 2},   // backbone segment
            {"A", 0.5,   2, 7},   // side chain 2, local_ds = 0.1
            {"B", 0.2,   2, 3},   // backbone segment
            {"A", 0.175, 3, 8},   // side chain 3, local_ds = 0.0875
            {"B", 0.2,   3, 4},   // backbone segment
            {"A", 0.35,  4, 9},   // side chain 4, local_ds = 0.0875
            {"B", 0.2,   4, 5},   // backbone segment
        };

        Molecules* molecules_brush = new Molecules("Continuous", 0.1, bond_lengths_brush);
        molecules_brush->add_polymer(1.0, blocks_brush, {});

        // Display polymer architecture
        molecules_brush->display_architectures();
        std::cout << std::endl;

        PropagatorComputationOptimizer* optimizer_brush =
            new PropagatorComputationOptimizer(molecules_brush, false);

        optimizer_brush->display_blocks();
        optimizer_brush->display_propagators();

        // Verify mapping
        const auto& mapping_brush = molecules_brush->get_contour_length_mapping();
        std::cout << std::endl;

        // Should have 5 unique lengths: 0.175, 0.2, 0.3, 0.35, 0.5
        int n_unique_lengths_brush = mapping_brush.get_n_unique_lengths();
        std::cout << "Number of unique contour lengths: " << n_unique_lengths_brush << std::endl;
        if (n_unique_lengths_brush != 5)
        {
            std::cout << "FAILED: Expected 5 unique lengths, got " << n_unique_lengths_brush << std::endl;
            return -1;
        }

        // Should have 2 unique local_ds values (0.0875 and 0.1)
        int n_unique_ds_brush = mapping_brush.get_n_unique_ds();
        std::cout << "Number of unique local_ds values: " << n_unique_ds_brush << std::endl;
        if (n_unique_ds_brush != 2)
        {
            std::cout << "FAILED: Expected 2 unique local_ds values, got " << n_unique_ds_brush << std::endl;
            return -1;
        }
        std::cout << "OK: Found 5 unique lengths and 2 unique local_ds values" << std::endl;

        // Verify n_segment values
        std::cout << std::endl;
        std::cout << "n_segment for 0.175: " << mapping_brush.get_n_segment(0.175) << " (expected 2)" << std::endl;
        std::cout << "n_segment for 0.2:   " << mapping_brush.get_n_segment(0.2) << " (expected 2)" << std::endl;
        std::cout << "n_segment for 0.3:   " << mapping_brush.get_n_segment(0.3) << " (expected 3)" << std::endl;
        std::cout << "n_segment for 0.35:  " << mapping_brush.get_n_segment(0.35) << " (expected 4)" << std::endl;
        std::cout << "n_segment for 0.5:   " << mapping_brush.get_n_segment(0.5) << " (expected 5)" << std::endl;

        // Verify ds_index sharing
        int ds_idx_brush_0175 = mapping_brush.get_ds_index(0.175);
        int ds_idx_brush_020 = mapping_brush.get_ds_index(0.2);
        int ds_idx_brush_030 = mapping_brush.get_ds_index(0.3);
        int ds_idx_brush_035 = mapping_brush.get_ds_index(0.35);
        int ds_idx_brush_050 = mapping_brush.get_ds_index(0.5);

        std::cout << std::endl;
        std::cout << "ds_index for 0.175: " << ds_idx_brush_0175 << std::endl;
        std::cout << "ds_index for 0.2:   " << ds_idx_brush_020 << std::endl;
        std::cout << "ds_index for 0.3:   " << ds_idx_brush_030 << std::endl;
        std::cout << "ds_index for 0.35:  " << ds_idx_brush_035 << std::endl;
        std::cout << "ds_index for 0.5:   " << ds_idx_brush_050 << std::endl;

        // 0.175 and 0.35 should share ds_index (local_ds = 0.0875)
        // 0.2, 0.3, and 0.5 should share ds_index (local_ds = 0.1)
        if (ds_idx_brush_0175 != ds_idx_brush_035)
        {
            std::cout << "FAILED: ds_index(0.175) and ds_index(0.35) should be equal" << std::endl;
            return -1;
        }
        if (ds_idx_brush_020 != ds_idx_brush_030 || ds_idx_brush_030 != ds_idx_brush_050)
        {
            std::cout << "FAILED: ds_index(0.2), ds_index(0.3), ds_index(0.5) should be equal" << std::endl;
            return -1;
        }
        if (ds_idx_brush_0175 == ds_idx_brush_020)
        {
            std::cout << "FAILED: ds_index(0.175) and ds_index(0.2) should differ" << std::endl;
            return -1;
        }
        std::cout << "OK: ds_index values correctly shared" << std::endl;

        // Test propagator computation with all solver variants:
        // - normal, aggregated, reduce_memory, reduce_memory+aggregated
        std::cout << std::endl;
        std::cout << "Testing propagator computation for bottle-brush with all solver variants..." << std::endl;

        // Create two optimizers: without and with aggregation
        PropagatorComputationOptimizer* optimizer_brush_agg =
            new PropagatorComputationOptimizer(molecules_brush, true);

        std::cout << std::endl << "Propagators with aggregation:" << std::endl;
        optimizer_brush_agg->display_propagators();

        std::vector<PropagatorComputation<double>*> solver_list_brush;
        std::vector<ComputationBox<double>*> cb_list_brush;
        std::vector<std::string> solver_name_list_brush;

        #ifdef USE_CPU_FFTW
        solver_name_list_brush.push_back("cpu-fftw");
        solver_name_list_brush.push_back("cpu-fftw, aggregated");
        solver_name_list_brush.push_back("cpu-fftw, reduce_memory");
        solver_name_list_brush.push_back("cpu-fftw, reduce_memory, aggregated");
        cb_list_brush.push_back(new CpuComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list_brush.push_back(new CpuComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list_brush.push_back(new CpuComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list_brush.push_back(new CpuComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list_brush.push_back(new CpuComputationContinuous<double>(
            cb_list_brush.end()[-4], molecules_brush, optimizer_brush, "pseudospectral"));
        solver_list_brush.push_back(new CpuComputationContinuous<double>(
            cb_list_brush.end()[-3], molecules_brush, optimizer_brush_agg, "pseudospectral"));
        solver_list_brush.push_back(new CpuComputationReduceMemoryContinuous<double>(
            cb_list_brush.end()[-2], molecules_brush, optimizer_brush, "pseudospectral"));
        solver_list_brush.push_back(new CpuComputationReduceMemoryContinuous<double>(
            cb_list_brush.end()[-1], molecules_brush, optimizer_brush_agg, "pseudospectral"));
        #endif

        #ifdef USE_CUDA
        solver_name_list_brush.push_back("cuda");
        solver_name_list_brush.push_back("cuda, aggregated");
        solver_name_list_brush.push_back("cuda, reduce_memory");
        solver_name_list_brush.push_back("cuda, reduce_memory, aggregated");
        cb_list_brush.push_back(new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list_brush.push_back(new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list_brush.push_back(new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        cb_list_brush.push_back(new CudaComputationBox<double>({II,JJ,KK}, {Lx,Ly,Lz}, {}));
        solver_list_brush.push_back(new CudaComputationContinuous<double>(
            cb_list_brush.end()[-4], molecules_brush, optimizer_brush, "pseudospectral"));
        solver_list_brush.push_back(new CudaComputationContinuous<double>(
            cb_list_brush.end()[-3], molecules_brush, optimizer_brush_agg, "pseudospectral"));
        solver_list_brush.push_back(new CudaComputationReduceMemoryContinuous<double>(
            cb_list_brush.end()[-2], molecules_brush, optimizer_brush, "pseudospectral"));
        solver_list_brush.push_back(new CudaComputationReduceMemoryContinuous<double>(
            cb_list_brush.end()[-1], molecules_brush, optimizer_brush_agg, "pseudospectral"));
        #endif

        // Store results for comparison
        std::vector<double> Q_values;
        std::vector<double> phi_a_sum_values;
        std::vector<double> phi_b_sum_values;
        std::vector<double> sum_phi_values;

        for(size_t n=0; n<solver_list_brush.size(); n++)
        {
            PropagatorComputation<double>* solver = solver_list_brush[n];

            std::cout << std::endl << "Running: " << solver_name_list_brush[n] << std::endl;

            solver->compute_propagators({{"A", w_a}, {"B", w_b}}, {});
            solver->compute_concentrations();

            double Q_brush = solver->get_total_partition(0);
            std::cout << "Partition function Q = " << Q_brush << std::endl;
            Q_values.push_back(Q_brush);

            // Get total concentration
            double phi_a_brush[M] = {0.0};
            double phi_b_brush[M] = {0.0};
            solver->get_total_concentration("A", phi_a_brush);
            solver->get_total_concentration("B", phi_b_brush);

            double sum_phi_a_brush = 0.0;
            double sum_phi_b_brush = 0.0;
            bool all_positive_brush = true;
            for(int i=0; i<M; i++)
            {
                sum_phi_a_brush += phi_a_brush[i];
                sum_phi_b_brush += phi_b_brush[i];
                if (phi_a_brush[i] < 0 || phi_b_brush[i] < 0)
                    all_positive_brush = false;
            }
            sum_phi_a_brush /= M;
            sum_phi_b_brush /= M;
            phi_a_sum_values.push_back(sum_phi_a_brush);
            phi_b_sum_values.push_back(sum_phi_b_brush);

            double sum_phi_brush = sum_phi_a_brush + sum_phi_b_brush;
            sum_phi_values.push_back(sum_phi_brush);
            std::cout << "Average total concentration: " << sum_phi_brush << std::endl;

            if (!all_positive_brush)
            {
                std::cout << "FAILED: Negative concentration detected" << std::endl;
                return -1;
            }
            if (std::abs(sum_phi_brush - 1.0) > 0.1)
            {
                std::cout << "FAILED: Average concentration should be ~1.0, got " << sum_phi_brush << std::endl;
                return -1;
            }
            std::cout << "OK: Propagator computation successful for " << solver_name_list_brush[n] << std::endl;
        }

        // Compare results across all solver variants
        std::cout << std::endl << "Comparing results across solver variants..." << std::endl;
        double Q_ref = Q_values[0];
        double phi_a_ref = phi_a_sum_values[0];
        double phi_b_ref = phi_b_sum_values[0];
        double sum_phi_ref = sum_phi_values[0];
        bool all_match = true;

        for(size_t n=1; n<Q_values.size(); n++)
        {
            double Q_diff = std::abs(Q_values[n] - Q_ref);
            double phi_a_diff = std::abs(phi_a_sum_values[n] - phi_a_ref);
            double phi_b_diff = std::abs(phi_b_sum_values[n] - phi_b_ref);
            double sum_phi_diff = std::abs(sum_phi_values[n] - sum_phi_ref);

            std::cout << solver_name_list_brush[n] << ": Q_diff=" << Q_diff
                      << ", phi_a_diff=" << phi_a_diff
                      << ", phi_b_diff=" << phi_b_diff
                      << ", sum_phi_diff=" << sum_phi_diff << std::endl;

            if (Q_diff > 1e-12 || phi_a_diff > 1e-12 || phi_b_diff > 1e-12 || sum_phi_diff > 1e-12)
            {
                std::cout << "FAILED: Results differ too much between variants" << std::endl;
                all_match = false;
            }
        }

        if (!all_match)
        {
            return -1;
        }
        std::cout << "OK: All solver variants produce consistent results" << std::endl;

        // Cleanup
        for(size_t n=0; n<solver_list_brush.size(); n++)
        {
            delete solver_list_brush[n];
            delete cb_list_brush[n];
        }
        delete optimizer_brush_agg;
        delete optimizer_brush;
        delete molecules_brush;

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
