/**
 * @file CudaComputationReduceMemoryGlobalRichardson.cu
 * @brief Memory-efficient GPU propagator computation with Global Richardson.
 *
 * This class implements memory-efficient Global Richardson extrapolation
 * using checkpointing to reduce GPU memory footprint.
 *
 * **Checkpointing Strategy:**
 *
 * Instead of storing all propagators, this implementation:
 * 1. Stores propagators at checkpoint intervals (every sqrt(N) steps)
 * 2. Maintains checkpoints for BOTH full-step and half-step chains
 * 3. Recomputes intermediate values on-the-fly during concentration calc
 * 4. Applies Richardson extrapolation during recomputation
 *
 * **Memory Layout:**
 *
 * - Checkpoints stored in pinned host memory
 * - Only a few GPU arrays needed for current computation
 * - Workspace arrays for recomputation
 *
 * @see CudaComputationGlobalRichardson for full-storage version
 * @see CpuComputationReduceMemoryGlobalRichardson for CPU version
 */

#include <cmath>
#include <cstring>
#include <omp.h>

#include "CudaComputationReduceMemoryGlobalRichardson.h"
#include "CudaComputationBox.h"
#include "SimpsonRule.h"
#include "PropagatorCode.h"

CudaComputationReduceMemoryGlobalRichardson::CudaComputationReduceMemoryGlobalRichardson(
    ComputationBox<double>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer)
    : PropagatorComputation<double>(cb, molecules, propagator_computation_optimizer),
      cb(cb), molecules(molecules),
      propagator_computation_optimizer(propagator_computation_optimizer)
{
    try
    {
        #ifndef NDEBUG
        std::cout << "--------- Global Richardson (Reduce Memory), CUDA ---------" << std::endl;
        #endif

        if (molecules->get_model_name() != "continuous")
            throw_with_line_number("Global Richardson only supports 'continuous' chain model.");

        const int M = cb->get_total_grid();

        // Single stream for reduce-memory mode
        n_streams = 1;
        gpu_error_check(cudaStreamCreate(&stream));

        // Create base solver
        solver = new CudaSolverGlobalRichardsonBase(cb, molecules, 1, nullptr);

        // Calculate checkpoint interval
        total_max_n_segment = 0;
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            total_max_n_segment += item.second.max_n_segment;
        }
        checkpoint_interval = (int)ceil(sqrt((double)total_max_n_segment));
        if (checkpoint_interval < 1)
            checkpoint_interval = 1;

        #ifndef NDEBUG
        std::cout << "Checkpoint interval: " << checkpoint_interval << std::endl;
        #endif

        // Allocate checkpoint propagators (in pinned host memory)
        if (propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("No propagator codes. Add polymers first.");

        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;

            // Number of checkpoints needed
            int n_checkpoints = max_n_segment / checkpoint_interval + 1;

            // Allocate checkpoint storage for full-step and half-step (pinned memory)
            for (int c = 0; c < n_checkpoints; c++)
            {
                double* h_full;
                double* h_half;
                gpu_error_check(cudaMallocHost((void**)&h_full, sizeof(double)*M));
                gpu_error_check(cudaMallocHost((void**)&h_half, sizeof(double)*M));
                d_propagator_full_at_check_point[std::make_tuple(key, c)] = h_full;
                d_propagator_half_at_check_point[std::make_tuple(key, c)] = h_half;
            }

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment + 1];
            for (int i = 0; i <= max_n_segment; i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate recomputation workspace (in pinned host memory)
        for (int i = 0; i <= checkpoint_interval; i++)
        {
            double* h_full;
            double* h_half;
            gpu_error_check(cudaMallocHost((void**)&h_full, sizeof(double)*M));
            gpu_error_check(cudaMallocHost((void**)&h_half, sizeof(double)*M));
            d_q_full_recal.push_back(h_full);
            d_q_half_recal.push_back(h_half);
        }

        // GPU ping-pong buffers
        gpu_error_check(cudaMalloc((void**)&d_q_full_pair[0], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_full_pair[1], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_half_pair[0], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_half_pair[1], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_half_temp, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_full_skip[0], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_full_skip[1], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_half_skip[0], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_half_skip[1], sizeof(double)*M));

        // Allocate concentration arrays (pinned host memory)
        if (propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("No blocks. Add polymers first.");

        for (const auto& item : propagator_computation_optimizer->get_computation_blocks())
        {
            double* h_phi;
            gpu_error_check(cudaMallocHost((void**)&h_phi, sizeof(double)*M));
            std::memset(h_phi, 0, sizeof(double)*M);
            d_phi_block[item.first] = h_phi;
        }

        // Set up partition function calculation info
        int current_p = 0;
        for (const auto& block : d_phi_block)
        {
            const auto& key = block.first;
            int p = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            if (p != current_p)
                continue;

            int n_aggregated = propagator_computation_optimizer->get_computation_block(key).v_u.size() /
                              propagator_computation_optimizer->get_computation_block(key).n_repeated;
            int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;

            partition_segment_info.push_back(std::make_tuple(
                p, key_left, key_right, n_segment_left, n_aggregated));
            current_p++;
        }

        // Solvent concentrations
        for (int s = 0; s < molecules->get_n_solvent_types(); s++)
        {
            double* h_phi;
            gpu_error_check(cudaMallocHost((void**)&h_phi, sizeof(double)*M));
            d_phi_solvent.push_back(h_phi);
        }

        // Create scheduler
        sc = new Scheduler(propagator_computation_optimizer->get_computation_propagators(), n_streams);

        // Allocate working arrays
        gpu_error_check(cudaMalloc((void**)&d_q_unity, sizeof(double)*M));
        double* h_unity = new double[M];
        for (int i = 0; i < M; i++)
            h_unity[i] = 1.0;
        gpu_error_check(cudaMemcpy(d_q_unity, h_unity, sizeof(double)*M, cudaMemcpyHostToDevice));
        delete[] h_unity;

        // Copy mask if present
        if (cb->get_mask() != nullptr)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_mask, sizeof(double)*M));
            gpu_error_check(cudaMemcpy(d_q_mask, cb->get_mask(), sizeof(double)*M, cudaMemcpyHostToDevice));
        }
        else
            d_q_mask = nullptr;

        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M));

        solver->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CudaComputationReduceMemoryGlobalRichardson::~CudaComputationReduceMemoryGlobalRichardson()
{
    delete solver;
    delete sc;

    // Free checkpoint propagators (pinned memory)
    for (const auto& item : d_propagator_full_at_check_point)
        cudaFreeHost(item.second);

    for (const auto& item : d_propagator_half_at_check_point)
        cudaFreeHost(item.second);

    // Free recomputation workspace (pinned memory)
    for (auto* ptr : d_q_full_recal)
        cudaFreeHost(ptr);

    for (auto* ptr : d_q_half_recal)
        cudaFreeHost(ptr);

    // Free GPU ping-pong buffers
    cudaFree(d_q_full_pair[0]);
    cudaFree(d_q_full_pair[1]);
    cudaFree(d_q_half_pair[0]);
    cudaFree(d_q_half_pair[1]);
    cudaFree(d_q_half_temp);
    cudaFree(d_q_full_skip[0]);
    cudaFree(d_q_full_skip[1]);
    cudaFree(d_q_half_skip[0]);
    cudaFree(d_q_half_skip[1]);

    // Free concentration arrays (pinned memory)
    for (const auto& item : d_phi_block)
        cudaFreeHost(item.second);

    for (const auto& item : d_phi_solvent)
        cudaFreeHost(item);

    #ifndef NDEBUG
    for (const auto& item : propagator_finished)
        delete[] item.second;
    #endif

    cudaFree(d_q_unity);
    if (d_q_mask != nullptr)
        cudaFree(d_q_mask);
    cudaFree(d_phi);

    cudaStreamDestroy(stream);
}

void CudaComputationReduceMemoryGlobalRichardson::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

void CudaComputationReduceMemoryGlobalRichardson::compute_propagators(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();

        // Validate inputs
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            if (w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" not in w_input.");
        }

        // Update Boltzmann factors (w_input is on host)
        solver->update_dw("cpu", w_input);

        #ifndef NDEBUG
        for (const auto& item : propagator_computation_optimizer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;
            for (int i = 0; i <= max_n_segment; i++)
                propagator_finished[key][i] = false;
        }
        #endif

        // Process propagators according to schedule (sequentially for reduce memory)
        auto& branch_schedule = sc->get_schedule();
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            for (size_t job = 0; job < parallel_job->size(); job++)
            {
                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = propagator_computation_optimizer->get_computation_propagator(key).deps;
                auto monomer_type = propagator_computation_optimizer->get_computation_propagator(key).monomer_type;

                // Initialize at segment 0
                if (n_segment_from == 0 && deps.size() == 0)
                {
                    // Leaf node
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpy(d_q_full_pair[0], q_init[g],
                            sizeof(double)*M, cudaMemcpyHostToDevice));
                        gpu_error_check(cudaMemcpy(d_q_half_pair[0], q_init[g],
                            sizeof(double)*M, cudaMemcpyHostToDevice));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpy(d_q_full_pair[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice));
                        gpu_error_check(cudaMemcpy(d_q_half_pair[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice));
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                else if (n_segment_from == 0 && deps.size() > 0)
                {
                    // Junction node
                    if (key[0] == '[')
                    {
                        // Aggregated (sum)
                        gpu_error_check(cudaMemset(d_q_full_pair[0], 0, sizeof(double)*M));
                        gpu_error_check(cudaMemset(d_q_half_pair[0], 0, sizeof(double)*M));

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            // Get from checkpoint
                            int checkpoint_idx = sub_n_segment / checkpoint_interval;
                            double* sub_full = d_propagator_full_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];
                            double* sub_half = d_propagator_half_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];

                            // Copy to temp buffers
                            gpu_error_check(cudaMemcpy(d_q_full_skip[0], sub_full, sizeof(double)*M, cudaMemcpyHostToDevice));
                            gpu_error_check(cudaMemcpy(d_q_half_skip[0], sub_half, sizeof(double)*M, cudaMemcpyHostToDevice));

                            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                d_q_full_pair[0], 1.0, d_q_full_pair[0],
                                sub_n_repeated, d_q_full_skip[0], M);

                            ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                d_q_half_pair[0], 1.0, d_q_half_pair[0],
                                sub_n_repeated, d_q_half_skip[0], M);
                        }
                        gpu_error_check(cudaPeekAtLastError());
                    }
                    else if (key[0] == '(')
                    {
                        // Product (junction)
                        gpu_error_check(cudaMemcpy(d_q_full_pair[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice));
                        gpu_error_check(cudaMemcpy(d_q_half_pair[0], d_q_unity,
                            sizeof(double)*M, cudaMemcpyDeviceToDevice));

                        for (size_t d = 0; d < deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment = std::get<1>(deps[d]);
                            int sub_n_repeated = std::get<2>(deps[d]);

                            // Get from checkpoint
                            int checkpoint_idx = sub_n_segment / checkpoint_interval;
                            double* sub_full = d_propagator_full_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];
                            double* sub_half = d_propagator_half_at_check_point[std::make_tuple(sub_dep, checkpoint_idx)];

                            // Copy to temp buffers
                            gpu_error_check(cudaMemcpy(d_q_full_skip[0], sub_full, sizeof(double)*M, cudaMemcpyHostToDevice));
                            gpu_error_check(cudaMemcpy(d_q_half_skip[0], sub_half, sizeof(double)*M, cudaMemcpyHostToDevice));

                            for (int r = 0; r < sub_n_repeated; r++)
                            {
                                ker_multi<<<N_BLOCKS, N_THREADS>>>(
                                    d_q_full_pair[0], d_q_full_pair[0],
                                    d_q_full_skip[0], 1.0, M);

                                ker_multi<<<N_BLOCKS, N_THREADS>>>(
                                    d_q_half_pair[0], d_q_half_pair[0],
                                    d_q_half_skip[0], 1.0, M);
                            }
                        }
                        gpu_error_check(cudaPeekAtLastError());
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                else
                {
                    // Starting from a checkpoint
                    int checkpoint_idx = n_segment_from / checkpoint_interval;
                    double* checkpoint_full = d_propagator_full_at_check_point[std::make_tuple(key, checkpoint_idx)];
                    double* checkpoint_half = d_propagator_half_at_check_point[std::make_tuple(key, checkpoint_idx)];

                    gpu_error_check(cudaMemcpy(d_q_full_pair[0], checkpoint_full, sizeof(double)*M, cudaMemcpyHostToDevice));
                    gpu_error_check(cudaMemcpy(d_q_half_pair[0], checkpoint_half, sizeof(double)*M, cudaMemcpyHostToDevice));
                }

                // Apply mask at segment 0
                if (n_segment_from == 0 && d_q_mask != nullptr)
                {
                    ker_multi<<<N_BLOCKS, N_THREADS>>>(
                        d_q_full_pair[0], d_q_full_pair[0], d_q_mask, 1.0, M);
                    ker_multi<<<N_BLOCKS, N_THREADS>>>(
                        d_q_half_pair[0], d_q_half_pair[0], d_q_mask, 1.0, M);
                    gpu_error_check(cudaPeekAtLastError());
                }

                // Store checkpoint at segment 0 if needed
                if (n_segment_from == 0)
                {
                    double* cp_full = d_propagator_full_at_check_point[std::make_tuple(key, 0)];
                    double* cp_half = d_propagator_half_at_check_point[std::make_tuple(key, 0)];
                    gpu_error_check(cudaMemcpy(cp_full, d_q_full_pair[0], sizeof(double)*M, cudaMemcpyDeviceToHost));
                    gpu_error_check(cudaMemcpy(cp_half, d_q_half_pair[0], sizeof(double)*M, cudaMemcpyDeviceToHost));
                }

                // Advance propagators
                int full_idx = 0;
                for (int n = n_segment_from; n < n_segment_to; n++)
                {
                    // Full-step chain: one full step
                    solver->advance_full_step(
                        0, d_q_full_pair[full_idx], d_q_full_pair[1 - full_idx], monomer_type, d_q_mask);

                    // Half-step chain: two half steps
                    solver->advance_half_step(
                        0, d_q_half_pair[full_idx], d_q_half_temp, monomer_type, d_q_mask);
                    solver->advance_half_step(
                        0, d_q_half_temp, d_q_half_pair[1 - full_idx], monomer_type, d_q_mask);

                    full_idx = 1 - full_idx;

                    // Store checkpoint if at checkpoint position
                    if ((n + 1) % checkpoint_interval == 0 || n + 1 == n_segment_to)
                    {
                        int checkpoint_idx = (n + 1) / checkpoint_interval;
                        double* cp_full = d_propagator_full_at_check_point[std::make_tuple(key, checkpoint_idx)];
                        double* cp_half = d_propagator_half_at_check_point[std::make_tuple(key, checkpoint_idx)];
                        gpu_error_check(cudaMemcpy(cp_full, d_q_full_pair[full_idx], sizeof(double)*M, cudaMemcpyDeviceToHost));
                        gpu_error_check(cudaMemcpy(cp_half, d_q_half_pair[full_idx], sizeof(double)*M, cudaMemcpyDeviceToHost));
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][n + 1] = true;
                    #endif
                }
            }
            gpu_error_check(cudaDeviceSynchronize());
        }

        // Compute partition functions using Richardson extrapolation
        for (const auto& segment_info : partition_segment_info)
        {
            int p = std::get<0>(segment_info);
            std::string key_left = std::get<1>(segment_info);
            std::string key_right = std::get<2>(segment_info);
            int n_segment_left = std::get<3>(segment_info);
            int n_aggregated = std::get<4>(segment_info);

            // Get checkpoints for q_left at n_segment_left and q_right at 0
            int checkpoint_left = n_segment_left / checkpoint_interval;
            int checkpoint_right = 0;

            double* q_full_left = d_propagator_full_at_check_point[std::make_tuple(key_left, checkpoint_left)];
            double* q_half_left = d_propagator_half_at_check_point[std::make_tuple(key_left, checkpoint_left)];
            double* q_full_right = d_propagator_full_at_check_point[std::make_tuple(key_right, checkpoint_right)];
            double* q_half_right = d_propagator_half_at_check_point[std::make_tuple(key_right, checkpoint_right)];

            // Compute Richardson-extrapolated inner product
            double q_inner_product = 0.0;
            for (int i = 0; i < M; i++)
            {
                double q_rich_left = (4.0 * q_half_left[i] - q_full_left[i]) / 3.0;
                double q_rich_right = (4.0 * q_half_right[i] - q_full_right[i]) / 3.0;
                q_inner_product += q_rich_left * q_rich_right;
            }
            q_inner_product *= (cb->get_volume() / M);

            single_polymer_partitions[p] = q_inner_product / (n_aggregated * cb->get_volume());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryGlobalRichardson::compute_concentrations()
{
    try
    {
        const int M = cb->get_total_grid();

        // Calculate segment concentrations with on-the-fly recomputation
        for (auto& block : d_phi_block)
        {
            const auto& key = block.first;

            int p = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            int n_segment_left = propagator_computation_optimizer->get_computation_block(key).n_segment_left;
            int n_repeated = propagator_computation_optimizer->get_computation_block(key).n_repeated;
            std::string monomer_type = propagator_computation_optimizer->get_computation_block(key).monomer_type;

            if (n_segment_right == 0)
            {
                std::memset(block.second, 0, sizeof(double)*M);
                continue;
            }

            // Compute φ with on-the-fly recomputation
            calculate_phi_one_block(
                block.second,
                key_left,
                key_right,
                n_segment_left,
                n_segment_right,
                monomer_type
            );

            // Normalize
            Polymer& pc = molecules->get_polymer(p);
            double norm = (molecules->get_ds() * pc.get_volume_fraction() / pc.get_alpha() * n_repeated) /
                         single_polymer_partitions[p];
            for (int i = 0; i < M; i++)
                block.second[i] *= norm;
        }

        // Solvent concentrations
        if (molecules->get_n_solvent_types() > 0)
        {
            throw_with_line_number("Solvent concentration not yet implemented for Global Richardson.");
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryGlobalRichardson::calculate_phi_one_block(
    double* d_phi_out,
    std::string key_left,
    std::string key_right,
    const int N_LEFT,
    const int N_RIGHT,
    std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();
        std::vector<double> simpson_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Initialize phi on device
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // Process in blocks defined by checkpoints
        int n_start = 0;
        while (n_start <= N_RIGHT)
        {
            int n_end = std::min(n_start + checkpoint_interval, N_RIGHT);

            // Find checkpoint for right propagator
            int checkpoint_right = n_start / checkpoint_interval;
            int right_checkpoint_pos = checkpoint_right * checkpoint_interval;

            // Find checkpoint for left propagator
            int left_pos_start = N_LEFT - n_start;
            int left_checkpoint = left_pos_start / checkpoint_interval;
            int left_checkpoint_pos = left_checkpoint * checkpoint_interval;

            // Load and advance right propagator from checkpoint to n_start
            double* cp_full_right = d_propagator_full_at_check_point[std::make_tuple(key_right, checkpoint_right)];
            double* cp_half_right = d_propagator_half_at_check_point[std::make_tuple(key_right, checkpoint_right)];

            gpu_error_check(cudaMemcpy(d_q_full_pair[0], cp_full_right, sizeof(double)*M, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_q_half_pair[0], cp_half_right, sizeof(double)*M, cudaMemcpyHostToDevice));

            // Advance from right_checkpoint_pos to n_start
            int full_idx = 0;
            for (int n = right_checkpoint_pos; n < n_start; n++)
            {
                solver->advance_full_step(0, d_q_full_pair[full_idx], d_q_full_pair[1 - full_idx], monomer_type, d_q_mask);
                solver->advance_half_step(0, d_q_half_pair[full_idx], d_q_half_temp, monomer_type, d_q_mask);
                solver->advance_half_step(0, d_q_half_temp, d_q_half_pair[1 - full_idx], monomer_type, d_q_mask);
                full_idx = 1 - full_idx;
            }

            // Compute and store right propagators for n_start to n_end
            gpu_error_check(cudaMemcpy(d_q_full_recal[0], d_q_full_pair[full_idx], sizeof(double)*M, cudaMemcpyDeviceToHost));
            gpu_error_check(cudaMemcpy(d_q_half_recal[0], d_q_half_pair[full_idx], sizeof(double)*M, cudaMemcpyDeviceToHost));

            for (int n = n_start; n < n_end; n++)
            {
                int local_idx = n - n_start;
                solver->advance_full_step(0, d_q_full_pair[full_idx], d_q_full_pair[1 - full_idx], monomer_type, d_q_mask);
                solver->advance_half_step(0, d_q_half_pair[full_idx], d_q_half_temp, monomer_type, d_q_mask);
                solver->advance_half_step(0, d_q_half_temp, d_q_half_pair[1 - full_idx], monomer_type, d_q_mask);
                full_idx = 1 - full_idx;
                gpu_error_check(cudaMemcpy(d_q_full_recal[local_idx + 1], d_q_full_pair[full_idx], sizeof(double)*M, cudaMemcpyDeviceToHost));
                gpu_error_check(cudaMemcpy(d_q_half_recal[local_idx + 1], d_q_half_pair[full_idx], sizeof(double)*M, cudaMemcpyDeviceToHost));
            }

            // Load checkpoint for left propagator
            double* cp_full_left = d_propagator_full_at_check_point[std::make_tuple(key_left, left_checkpoint)];
            double* cp_half_left = d_propagator_half_at_check_point[std::make_tuple(key_left, left_checkpoint)];

            gpu_error_check(cudaMemcpy(d_q_full_skip[0], cp_full_left, sizeof(double)*M, cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_q_half_skip[0], cp_half_left, sizeof(double)*M, cudaMemcpyHostToDevice));

            // Advance from left_checkpoint_pos to left_pos_start
            int skip_idx = 0;
            for (int n = left_checkpoint_pos; n < left_pos_start; n++)
            {
                solver->advance_full_step(0, d_q_full_skip[skip_idx], d_q_full_skip[1 - skip_idx], monomer_type, d_q_mask);
                solver->advance_half_step(0, d_q_half_skip[skip_idx], d_q_half_temp, monomer_type, d_q_mask);
                solver->advance_half_step(0, d_q_half_temp, d_q_half_skip[1 - skip_idx], monomer_type, d_q_mask);
                skip_idx = 1 - skip_idx;
            }

            // Accumulate phi contributions with Richardson extrapolation
            for (int n = n_start; n <= n_end; n++)
            {
                int local_right = n - n_start;

                // Copy right propagators to device
                gpu_error_check(cudaMemcpy(d_q_full_pair[0], d_q_full_recal[local_right], sizeof(double)*M, cudaMemcpyHostToDevice));
                gpu_error_check(cudaMemcpy(d_q_half_pair[0], d_q_half_recal[local_right], sizeof(double)*M, cudaMemcpyHostToDevice));

                // Richardson extrapolation: q_rich = (4*q_half - q_full) / 3
                // phi += coeff * q_rich_left * q_rich_right
                // = coeff * (4*q_half_left - q_full_left)/3 * (4*q_half_right - q_full_right)/3
                // = coeff/9 * (16*q_half_left*q_half_right - 4*q_half_left*q_full_right - 4*q_full_left*q_half_right + q_full_left*q_full_right)

                double coeff = simpson_coeff[n];

                // Compute: phi += coeff * q_rich_left * q_rich_right
                // Using kernel: phi = phi + coeff/9 * (16*half_l*half_r - 4*half_l*full_r - 4*full_l*half_r + full_l*full_r)
                // d_q_full_skip[skip_idx] = full_left, d_q_half_skip[skip_idx] = half_left
                // d_q_full_pair[0] = full_right, d_q_half_pair[0] = half_right

                ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_half_skip[skip_idx], d_q_half_pair[0], coeff * 16.0 / 9.0, M);
                ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_half_skip[skip_idx], d_q_full_pair[0], -coeff * 4.0 / 9.0, M);
                ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_full_skip[skip_idx], d_q_half_pair[0], -coeff * 4.0 / 9.0, M);
                ker_add_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_full_skip[skip_idx], d_q_full_pair[0], coeff * 1.0 / 9.0, M);
                gpu_error_check(cudaPeekAtLastError());

                // Advance left propagator
                if (n < n_end)
                {
                    solver->advance_full_step(0, d_q_full_skip[skip_idx], d_q_full_skip[1 - skip_idx], monomer_type, d_q_mask);
                    solver->advance_half_step(0, d_q_half_skip[skip_idx], d_q_half_temp, monomer_type, d_q_mask);
                    solver->advance_half_step(0, d_q_half_temp, d_q_half_skip[1 - skip_idx], monomer_type, d_q_mask);
                    skip_idx = 1 - skip_idx;
                }
            }

            n_start = n_end + 1;
        }

        // Copy result to output (pinned host memory)
        gpu_error_check(cudaMemcpy(d_phi_out, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryGlobalRichardson::compute_statistics(
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    compute_propagators(w_input, q_init);
    compute_concentrations();
}

void CudaComputationReduceMemoryGlobalRichardson::compute_stress()
{
    throw_with_line_number("Stress computation not yet implemented for Global Richardson.");
}

void CudaComputationReduceMemoryGlobalRichardson::get_chain_propagator(
    double* q_out, int polymer, int v, int u, int n)
{
    throw_with_line_number("get_chain_propagator requires recomputation - not yet implemented for reduce memory.");
}

void CudaComputationReduceMemoryGlobalRichardson::get_total_concentration(
    int p, std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        for (const auto& block : d_phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                for (int i = 0; i < M; i++)
                    phi[i] += block.second[i];
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryGlobalRichardson::get_total_concentration_gce(
    double fugacity, int p, std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P - 1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P - 1) + "]");

        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        for (const auto& block : d_phi_block)
        {
            int polymer_idx = std::get<0>(block.first);
            std::string key_left = std::get<1>(block.first);
            int n_segment_right = propagator_computation_optimizer->get_computation_block(block.first).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                Polymer& pc = molecules->get_polymer(p);
                double norm = fugacity / pc.get_volume_fraction() * pc.get_alpha() * single_polymer_partitions[p];
                for (int i = 0; i < M; i++)
                    phi[i] += block.second[i] * norm;
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryGlobalRichardson::get_total_concentration(
    std::string monomer_type, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();

        for (int i = 0; i < M; i++)
            phi[i] = 0.0;

        for (const auto& block : d_phi_block)
        {
            const auto& key = block.first;
            std::string block_monomer_type = propagator_computation_optimizer->get_computation_block(key).monomer_type;

            if (block_monomer_type == monomer_type)
            {
                for (int i = 0; i < M; i++)
                    phi[i] += block.second[i];
            }
        }

        for (int s = 0; s < molecules->get_n_solvent_types(); s++)
        {
            std::string solvent_monomer_type = std::get<1>(molecules->get_solvent(s));
            if (solvent_monomer_type == monomer_type)
            {
                for (int i = 0; i < M; i++)
                    phi[i] += d_phi_solvent[s][i];
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaComputationReduceMemoryGlobalRichardson::get_solvent_concentration(int s, double* phi)
{
    try
    {
        const int M = cb->get_total_grid();

        if (s < 0 || s >= molecules->get_n_solvent_types())
            throw_with_line_number("Invalid solvent index: " + std::to_string(s));

        for (int i = 0; i < M; i++)
            phi[i] = d_phi_solvent[s][i];
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

bool CudaComputationReduceMemoryGlobalRichardson::check_total_partition()
{
    int n_polymer_types = molecules->get_n_polymer_types();

    std::cout << "Global Richardson (Reduce Memory, CUDA) Partition Functions:" << std::endl;
    std::cout << "Polymer\tQ_richardson" << std::endl;

    for (int p = 0; p < n_polymer_types; p++)
    {
        std::cout << p << "\t" << single_polymer_partitions[p] << std::endl;
    }

    return true;
}
