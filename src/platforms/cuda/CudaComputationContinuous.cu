#include <complex>
#include <thrust/reduce.h>
#include <omp.h>
#include "CudaComputationContinuous.h"
#include "CudaComputationBox.h"
#include "CudaSolverPseudo.h"
#include "CudaSolverReal.h"
#include "SimpsonRule.h"

CudaComputationContinuous::CudaComputationContinuous(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorAnalyzer *propagator_analyzer,
    std::string method)
    : PropagatorComputation(cb, molecules, propagator_analyzer)
{
    try{
        const int M = cb->get_n_grid();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        // The number of parallel streams for propagator computation
        const char *ENV_OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
        std::string env_omp_num_threads(ENV_OMP_NUM_THREADS ? ENV_OMP_NUM_THREADS  : "");
        if (env_omp_num_threads.empty())
            n_streams = MAX_STREAMS;
        else
            n_streams =  std::min(std::stoi(env_omp_num_threads), MAX_STREAMS);
        std::cout << "n_streams: " << n_streams << std::endl;

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaSetDevice(i % N_GPUS));
            gpu_error_check(cudaStreamCreate(&streams[i][0])); // for kernel execution
            gpu_error_check(cudaStreamCreate(&streams[i][1])); // for memcpy
        }

        this->method = method;
        if(method == "pseudospectral")
            this->propagator_solver = new CudaSolverPseudo(cb, molecules, n_streams, streams, false);
        else if(method == "realspace")
            this->propagator_solver = new CudaSolverReal(cb, molecules, n_streams, streams, false);

        // Allocate memory for propagators
        gpu_error_check(cudaSetDevice(0));
        if( propagator_analyzer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        for(const auto& item: propagator_analyzer->get_computation_propagators())
        {
            std::string key = item.first;
            int max_n_segment = item.second.max_n_segment;
            
            propagator_size[key] = max_n_segment+1;
            d_propagator[key] = new double*[max_n_segment+1];
            for(int i=0; i<propagator_size[key]; i++)
                gpu_error_check(cudaMalloc((void**)&d_propagator[key][i], sizeof(double)*M));

            #ifndef NDEBUG
            propagator_finished[key] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                propagator_finished[key][i] = false;
            #endif
        }

        // Allocate memory for concentrations
        if( propagator_analyzer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");
        for(const auto& item: propagator_analyzer->get_computation_blocks())
        {
            d_phi_block[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_phi_block[item.first], sizeof(double)*M));
        }
        
        // Total partition functions for each polymer
        single_polymer_partitions = new double[molecules->get_n_polymer_types()];

        // Remember one segment for each polymer chain to compute total partition function
        int current_p = 0;
        for(const auto& block: d_phi_block)
        {
            const auto& key = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            // Skip if already found one segment
            if (p != current_p)
                continue;

            int n_aggregated = propagator_analyzer->get_computation_block(key).v_u.size()/
                               propagator_analyzer->get_computation_block(key).n_repeated;
            int n_segment_left = propagator_analyzer->get_computation_block(key).n_segment_left;

            single_partition_segment.push_back(std::make_tuple(
                p,
                d_propagator[key_left][n_segment_left],   // q
                d_propagator[key_right][0],               // q_dagger
                n_aggregated                              // how many propagators are aggregated
                ));
            current_p++;
        }

        // Total partition functions for each solvent
        single_solvent_partitions = new double[molecules->get_n_solvent_types()];

        // Concentrations for each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
        {
            double *d_phi_;
            gpu_error_check(cudaMalloc((void**)&d_phi_, sizeof(double)*M));
            d_phi_solvent.push_back(d_phi_);
        }

        // Create scheduler for computation of propagator
        sc = new Scheduler(propagator_analyzer->get_computation_propagators(), n_streams); 

        // Allocate memory for pseudo-spectral: advance_propagator()
        double q_unity[M];
        for(int i=0; i<M; i++)
            q_unity[i] = 1.0;
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_q_mask[gpu], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_q_unity[gpu], sizeof(double)*M));
            gpu_error_check(cudaMemcpy(d_q_unity[gpu], q_unity, sizeof(double)*M, cudaMemcpyHostToDevice));
        }

        for(int gpu=1; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_propagator_device[gpu][0], sizeof(double)*M));  // prev
            gpu_error_check(cudaMalloc((void**)&d_propagator_device[gpu][1], sizeof(double)*M));  // next
        }

        gpu_error_check(cudaSetDevice(0));
        gpu_error_check(cudaMalloc((void**)&d_phi,           sizeof(double)*M));

        // Allocate memory for stress calculation: compute_stress()
        for(int i=0; i<n_streams; i++)
        {
            gpu_error_check(cudaSetDevice(i % N_GPUS));
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][0], sizeof(double)*2*M)); // prev
            gpu_error_check(cudaMalloc((void**)&d_q_pair[i][1], sizeof(double)*2*M)); // next
        }

        // Copy mask to d_q_mask
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            if (cb->get_mask() != nullptr)
            {
                gpu_error_check(cudaMemcpy(d_q_mask[gpu], cb->get_mask(), sizeof(double)*M, cudaMemcpyHostToDevice));
            }
            else
            {
                d_q_mask[gpu] = nullptr;
            }
        }

        propagator_solver->update_laplacian_operator();
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaComputationContinuous::~CudaComputationContinuous()
{
    const int N_GPUS = CudaCommon::get_instance().get_n_gpus();
    
    delete propagator_solver;
    delete sc;

    delete[] single_polymer_partitions;
    delete[] single_solvent_partitions;

    for(const auto& item: d_propagator)
    {
        for(int i=0; i<propagator_size[item.first]; i++)
            cudaFree(item.second[i]);
        delete[] item.second;
    }
    for(const auto& item: d_phi_block)
        cudaFree(item.second);
    for(const auto& item: d_phi_solvent)
        cudaFree(item);

    #ifndef NDEBUG
    for(const auto& item: propagator_finished)
        delete[] item.second;
    #endif

    cudaFree(d_phi);

    // For pseudo-spectral: advance_propagator()
    for(int gpu=0; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_q_mask[gpu]);
        cudaFree(d_q_unity[gpu]);
    }
    
    for(int gpu=1; gpu<N_GPUS; gpu++)
    {
        cudaFree(d_propagator_device[gpu][0]);
        cudaFree(d_propagator_device[gpu][1]);
    }

    // For stress calculation: compute_stress()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_q_pair[i][0]);
        cudaFree(d_q_pair[i][1]);
    }

    // Destroy streams
    for(int i=0; i<n_streams; i++)
    {
        cudaStreamDestroy(streams[i][0]);
        cudaStreamDestroy(streams[i][1]);
    }
}

void CudaComputationContinuous::update_laplacian_operator()
{
    try{
        propagator_solver->update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
void CudaComputationContinuous::compute_statistics(
    std::string device,
    std::map<std::string, const double*> w_input,
    std::map<std::string, const double*> q_init)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int M = cb->get_n_grid();
        const double ds = molecules->get_ds();

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        for(const auto& item: propagator_analyzer->get_computation_propagators())
        {
            std::string monomer_type = item.second.monomer_type.substr(0, 1);
            if( w_input.find(monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in w_input.");
        }

        // Update dw or d_exp_dw
        propagator_solver->update_dw(device, w_input);
       
        auto& branch_schedule = sc->get_schedule();

        // For each time span
        for (auto parallel_job = branch_schedule.begin(); parallel_job != branch_schedule.end(); parallel_job++)
        {
            // For each propagator
            #pragma omp parallel for num_threads(n_streams)
            for(size_t job=0; job<parallel_job->size(); job++)
            {
                const int STREAM = omp_get_thread_num();
                int gpu = omp_get_thread_num() % N_GPUS;
                gpu_error_check(cudaSetDevice(gpu));

                // printf("gpu, STREAM: %d, %d\n ", gpu, STREAM);

                auto& key = std::get<0>((*parallel_job)[job]);
                int n_segment_from = std::get<1>((*parallel_job)[job]);
                int n_segment_to = std::get<2>((*parallel_job)[job]);
                auto& deps = propagator_analyzer->get_computation_propagator(key).deps;
                auto monomer_type = propagator_analyzer->get_computation_propagator(key).monomer_type.substr(0, 1);

                // Check key
                #ifndef NDEBUG
                if (d_propagator.find(key) == d_propagator.end())
                    throw_with_line_number("Could not find key '" + key + "'. ");
                #endif

                double **_d_propagator = d_propagator[key];

                // If it is leaf node
                if(n_segment_from == 1 && deps.size() == 0)
                {
                     // q_init
                    if (key[0] == '{')
                    {
                        std::string g = PropagatorCode::get_q_input_idx_from_key(key);
                        if (q_init.find(g) == q_init.end())
                            throw_with_line_number("Could not find q_init[\"" + g + "\"].");
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[0], q_init[g],
                            sizeof(double)*M, cudaMemcpyInputToDevice, streams[STREAM][0]));
                    }
                    else
                    {
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[0], d_q_unity[0],
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));
                    }

                    #ifndef NDEBUG
                    propagator_finished[key][0] = true;
                    #endif
                }
                // If it is not leaf node
                else if (n_segment_from == 1 && deps.size() > 0)
                {
                    // If it is aggregated
                    if (key[0] == '[')
                    {
                        // Initialize to zero
                        gpu_error_check(cudaMemsetAsync(_d_propagator[0], 0, sizeof(double)*M, streams[STREAM][0]));

                        // Add all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);
                            int sub_n_repeated  = std::get<2>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (d_propagator.find(sub_dep) == d_propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                _d_propagator[0], 1.0, _d_propagator[0],
                                sub_n_repeated, d_propagator[sub_dep][sub_n_segment], M);
                        }

                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                    else
                    {
                        // Initialize to one
                        gpu_error_check(cudaMemcpyAsync(_d_propagator[0], d_q_unity[0],
                            sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][0]));

                        // Multiply all propagators at junction if necessary 
                        for(size_t d=0; d<deps.size(); d++)
                        {
                            std::string sub_dep = std::get<0>(deps[d]);
                            int sub_n_segment   = std::get<1>(deps[d]);

                            // Check sub key
                            #ifndef NDEBUG
                            if (d_propagator.find(sub_dep) == d_propagator.end())
                                throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                            if (!propagator_finished[sub_dep][sub_n_segment])
                                throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                            #endif

                            multi_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                                _d_propagator[0], _d_propagator[0],
                                d_propagator[sub_dep][sub_n_segment], 1.0, M);
                        }
                        
                        #ifndef NDEBUG
                        propagator_finished[key][0] = true;
                        #endif
                    }
                }

                // Multiply mask
                if (n_segment_from == 1 && d_q_mask[0] != nullptr)
                    multi_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(_d_propagator[0], _d_propagator[0], d_q_mask[0], 1.0, M);

                if (gpu == 0)
                {
                    for(int n=n_segment_from; n<=n_segment_to; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[key][n-1])
                            throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                        #endif

                        // STREAM 0
                        propagator_solver->advance_propagator_continuous(
                            gpu, STREAM, 
                            _d_propagator[n-1],
                            _d_propagator[n],
                            monomer_type, d_q_mask[gpu]);

                        #ifndef NDEBUG
                        propagator_finished[key][n] = true;
                        #endif
                    }
                }
                else if(gpu >= 1)
                {
                    int prev, next;
                    prev = 0;
                    next = 1;

                    // Create events
                    cudaEvent_t kernel_done;
                    cudaEvent_t memcpy_done;
                    gpu_error_check(cudaEventCreate(&kernel_done));
                    gpu_error_check(cudaEventCreate(&memcpy_done));

                    // Wait until initialization process is done
                    gpu_error_check(cudaEventRecord(kernel_done, streams[STREAM][0]));
                    gpu_error_check(cudaStreamWaitEvent(streams[STREAM][1], kernel_done, 0));

                    // Copy propagator copy memory from device 1 to device
                    gpu_error_check(cudaMemcpyAsync(
                        d_propagator_device[gpu][prev],
                        _d_propagator[n_segment_from-1],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][1]));

                    gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                    gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

                    for(int n=n_segment_from; n<=n_segment_to; n++)
                    {
                        #ifndef NDEBUG
                        if (!propagator_finished[key][n-1])
                            throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                        #endif

                        // DEVICE 1, STREAM 0: calculate propagators
                        propagator_solver->advance_propagator_continuous(
                            gpu, STREAM, 
                            d_propagator_device[gpu][prev],
                            d_propagator_device[gpu][next],
                            monomer_type, d_q_mask[gpu]);
                        gpu_error_check(cudaEventRecord(kernel_done, streams[STREAM][0]));

                        // DEVICE 1, STREAM 1: copy memory from device 1 to device 0
                        if (n > n_segment_from)
                        {
                            gpu_error_check(cudaMemcpyAsync(
                                _d_propagator[n-1],
                                d_propagator_device[gpu][prev],
                                sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][1]));
                            gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                        }

                        // Wait until computation and memory copy are done
                        gpu_error_check(cudaStreamWaitEvent(streams[STREAM][1], kernel_done, 0));
                        gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

                        std::swap(prev, next);

                        #ifndef NDEBUG
                        propagator_finished[key][n] = true;
                        #endif
                    }

                    // Copy memory from device 1 to device 0
                    gpu_error_check(cudaMemcpyAsync(
                        _d_propagator[n_segment_to],
                        d_propagator_device[gpu][prev],
                        sizeof(double)*M, cudaMemcpyDeviceToDevice, streams[STREAM][1]));

                    gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                    gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));
                
                    gpu_error_check(cudaEventDestroy(kernel_done));
                    gpu_error_check(cudaEventDestroy(memcpy_done));
                }
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
                gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
            }
            // Synchronize all GPUs
            for(int gpu=0; gpu<N_GPUS; gpu++)
            {
                gpu_error_check(cudaSetDevice(gpu));
                gpu_error_check(cudaDeviceSynchronize());
            }
        }

        gpu_error_check(cudaSetDevice(0));
        // Compute total partition function of each distinct polymers
        for(const auto& segment_info: single_partition_segment)
        {
            int p                      = std::get<0>(segment_info);
            double *d_propagator_left  = std::get<1>(segment_info);
            double *d_propagator_right = std::get<2>(segment_info);
            int n_aggregated           = std::get<3>(segment_info);

            single_polymer_partitions[p] = cb->inner_product_device(
                d_propagator_left, d_propagator_right)/n_aggregated/cb->get_volume();
        }

        // Calculate segment concentrations
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            int n_segment_right = propagator_analyzer->get_computation_block(key).n_segment_right;
            int n_segment_left  = propagator_analyzer->get_computation_block(key).n_segment_left;
            int n_repeated = propagator_analyzer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(n_segment_right == 0)
            {
                gpu_error_check(cudaMemset(d_block.second, 0, sizeof(double)*M));
                continue;
            }

            // Check keys
            #ifndef NDEBUG
            if (d_propagator.find(key_left) == d_propagator.end())
                throw_with_line_number("Could not find key_left key'" + key_left + "'. ");
            if (d_propagator.find(key_right) == d_propagator.end())
                throw_with_line_number("Could not find key_right key'" + key_right + "'. ");
            #endif

            // Calculate phi of one block (possibly multiple blocks when using aggregation)
            calculate_phi_one_block(
                d_block.second,           // phi
                d_propagator[key_left],   // dependency v
                d_propagator[key_right],  // dependency u
                n_segment_right,
                n_segment_left);

            // Normalize concentration
            Polymer& pc = molecules->get_polymer(p);
            double norm = molecules->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p]*n_repeated;
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_block.second, norm, d_block.second, 0.0, d_block.second, M);
        }

        // Calculate partition functions and concentrations of solvents
        for(int s=0; s<molecules->get_n_solvent_types(); s++)
        {
            double *d_phi_ = d_phi_solvent[s];
            double volume_fraction = std::get<0>(molecules->get_solvent(s));
            std::string monomer_type = std::get<1>(molecules->get_solvent(s)).substr(0, 1);
            double *_d_exp_dw = propagator_solver->d_exp_dw[0][monomer_type];

            single_solvent_partitions[s] = cb->inner_product_device(_d_exp_dw, _d_exp_dw)/cb->get_volume();
            multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi_,_d_exp_dw, _d_exp_dw, volume_fraction/single_solvent_partitions[s], M);
        }
        gpu_error_check(cudaSetDevice(0));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationContinuous::calculate_phi_one_block(
    double *d_phi, double **d_q_1, double **d_q_2, const int N_RIGHT, const int N_LEFT)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonRule::get_coeff(N_RIGHT);

        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_LEFT], d_q_2[0], simpson_rule_coeff[0], M);
        for(int n=1; n<=N_RIGHT; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, d_q_1[N_LEFT-n], d_q_2[n], simpson_rule_coeff[n], M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaComputationContinuous::get_total_partition(int polymer)
{
    try
    {
        return single_polymer_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationContinuous::get_total_concentration(std::string monomer_type, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_n_grid();

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            std::string key_left = std::get<1>(key);
            int n_segment_right = propagator_analyzer->get_computation_block(key).n_segment_right;
            if (PropagatorCode::get_monomer_type_from_key(key_left).substr(0,1) == monomer_type && n_segment_right != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }

        // For each solvent
        for(int s=0;s<molecules->get_n_solvent_types();s++)
        {
            if (std::get<1>(molecules->get_solvent(s)) == monomer_type)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_phi_solvent[s], M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationContinuous::get_total_concentration(int p, std::string monomer_type, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        // For each block
        for(const auto& d_block: d_phi_block)
        {
            const auto& key = d_block.first;
            int polymer_idx = std::get<0>(key);
            std::string key_left = std::get<1>(key);
            int n_segment_right = propagator_analyzer->get_computation_block(key).n_segment_right;
            if (polymer_idx == p && PropagatorCode::get_monomer_type_from_key(key_left).substr(0,1) == monomer_type && n_segment_right != 0)
                lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, 1.0, d_block.second, M);
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationContinuous::get_block_concentration(int p, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int P = molecules->get_n_polymer_types();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (propagator_analyzer->is_aggregated())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        // Initialize to zero
        gpu_error_check(cudaMemset(d_phi, 0, sizeof(double)*M));

        Polymer& pc = molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, 1.0, d_phi_block[std::make_tuple(p, key_left, key_right)], M);
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaComputationContinuous::get_solvent_partition(int s)
{
    try
    {
        return single_solvent_partitions[s];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationContinuous::get_solvent_concentration(int s, double *phi)
{
    try
    {
        gpu_error_check(cudaSetDevice(0));
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int S = molecules->get_n_solvent_types();

        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");

        gpu_error_check(cudaMemcpy(phi, d_phi_solvent[s], sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaComputationContinuous::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        if (this->method == "realspace")
            throw_with_line_number("Currently, the real-space method does not support stress computation.");

        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        const int DIM = cb->get_dim();
        const int M   = cb->get_n_grid();

        std::vector<double> stress(DIM);
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> block_dq_dl[n_streams];

        // Reset stress map
        for(const auto& item: d_phi_block)
        {
            for(int i=0; i<n_streams; i++)
                for(int d=0; d<3; d++)
                    block_dq_dl[i][item.first][d] = 0.0;
        }

        // Compute stress for each block
        #pragma omp parallel for num_threads(n_streams)
        for(size_t b=0; b<d_phi_block.size();b++)
        {
            const int STREAM = omp_get_thread_num();
            int gpu = omp_get_thread_num() % N_GPUS;

            gpu_error_check(cudaSetDevice(gpu));

            auto block = d_phi_block.begin();
            advance(block, b);
            const auto& key   = block->first;

            // printf("start, b, gpu, STREAM: %2d, %2d, %2d\n", b, gpu, STREAM);

            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);

            const int N_RIGHT = propagator_analyzer->get_computation_block(key).n_segment_right;
            const int N_LEFT  = propagator_analyzer->get_computation_block(key).n_segment_left;
            std::string monomer_type = propagator_analyzer->get_computation_block(key).monomer_type.substr(0,1);
            int n_repeated = propagator_analyzer->get_computation_block(key).n_repeated;

            // If there is no segment
            if(N_RIGHT == 0)
                continue;

            // std::cout << p << ", " << key_left << ", " << key_right << ", " << N << ", " << N_LEFT << std::endl;

            std::vector<double> s_coeff = SimpsonRule::get_coeff(N_RIGHT);
            double** d_q_1 = d_propagator[key_left];     // dependency v
            double** d_q_2 = d_propagator[key_right];    // dependency u

            std::array<double,3> _block_dq_dl = {0.0, 0.0, 0.0};
            
            double *d_segment_stress;
            double segment_stress[DIM];
            gpu_error_check(cudaMalloc((void**)&d_segment_stress, sizeof(double)*3));

            int prev, next;
            prev = 0;
            next = 1;

            // Create events
            cudaEvent_t kernel_done;
            cudaEvent_t memcpy_done;
            gpu_error_check(cudaEventCreate(&kernel_done));
            gpu_error_check(cudaEventCreate(&memcpy_done));

            gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][prev][0], d_q_1[N_LEFT],
                    sizeof(double)*M,cudaMemcpyDeviceToDevice, streams[STREAM][1]));
            gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][prev][M], d_q_2[0],
                    sizeof(double)*M,cudaMemcpyDeviceToDevice, streams[STREAM][1]));

            gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
            gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

            for(int n=0; n<=N_RIGHT; n++)
            {
                // STREAM 1: Copy data
                if (n+1 <= N_RIGHT)
                {
                    gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][next][0], d_q_1[N_LEFT-n-1],
                            sizeof(double)*M,cudaMemcpyDeviceToDevice, streams[STREAM][1]));
                    gpu_error_check(cudaMemcpyAsync(&d_q_pair[STREAM][next][M], d_q_2[n+1],
                            sizeof(double)*M,cudaMemcpyDeviceToDevice, streams[STREAM][1]));
                    gpu_error_check(cudaEventRecord(memcpy_done, streams[STREAM][1]));
                }

                // STREAM 0: Compute stress
                propagator_solver->compute_single_segment_stress_continuous(
                    gpu, STREAM, d_q_pair[STREAM][prev], d_segment_stress, monomer_type);   
                gpu_error_check(cudaEventRecord(kernel_done, streams[STREAM][0]));

                // Wait until computation and memory copy are done
                gpu_error_check(cudaStreamWaitEvent(streams[STREAM][1], kernel_done, 0));
                gpu_error_check(cudaStreamWaitEvent(streams[STREAM][0], memcpy_done, 0));

                gpu_error_check(cudaMemcpy(segment_stress, d_segment_stress, sizeof(double)*DIM, cudaMemcpyDeviceToHost));
                for(int d=0; d<DIM; d++)
                    _block_dq_dl[d] += segment_stress[d]*s_coeff[n]*n_repeated;

                std::swap(prev, next);
            }
            gpu_error_check(cudaStreamSynchronize(streams[STREAM][0]));
            gpu_error_check(cudaStreamSynchronize(streams[STREAM][1]));
            gpu_error_check(cudaEventDestroy(kernel_done));
            gpu_error_check(cudaEventDestroy(memcpy_done));

            // std::cout << p << ", " << key_left << ", " << key_right << ", " << N << ", " << N_LEFT << std::endl;
            // std::cout << "STREAM, _block_dq_dl[0] " << STREAM  << ", " << _block_dq_dl[0] << std::endl;

            for(int d=0; d<DIM; d++)
                block_dq_dl[STREAM][key][d] += _block_dq_dl[d];
                
            cudaFree(d_segment_stress);
        }
        // Synchronize all GPUs
        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaDeviceSynchronize());
        }

        gpu_error_check(cudaSetDevice(0));
        // Compute total stress
        for(int d=0; d<DIM; d++)
            stress[d] = 0.0;
        for(const auto& block: d_phi_block)
        {
            const auto& key = block.first;
            int p                 = std::get<0>(key);
            std::string key_left  = std::get<1>(key);
            std::string key_right = std::get<2>(key);
            Polymer& pc  = molecules->get_polymer(p);

            for(int i=0; i<n_streams; i++)
                for(int d=0; d<DIM; d++)
                    stress[d] += block_dq_dl[i][key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_polymer_partitions[p];
        }
        for(int d=0; d<DIM; d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/molecules->get_ds();
            
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaComputationContinuous::get_chain_propagator(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get chain propagator for a selected polymer, block and direction.
    // This is made for debugging and testing.
    try
    {
        const int M = cb->get_n_grid();
        Polymer& pc = molecules->get_polymer(polymer);
        std::string dep = pc.get_propagator_key(v,u);

        // if (propagator_analyzer->get_computation_propagators().find(dep) == propagator_analyzer->get_computation_propagators().end())
        //     throw_with_line_number("Could not find the propagator code '" + dep + "'. Disable 'aggregation' option to obtain propagator_analyzer.");

        const int N_RIGHT = propagator_analyzer->get_computation_propagator(dep).max_n_segment;
        if (n < 0 || n > N_RIGHT)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N_RIGHT) + "]");

        gpu_error_check(cudaMemcpy(q_out, d_propagator[dep][n], sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
bool CudaComputationContinuous::check_total_partition()
{
    const int M = cb->get_n_grid();
    int n_polymer_types = molecules->get_n_polymer_types();
    std::vector<std::vector<double>> total_partitions;
    for(int p=0;p<n_polymer_types;p++)
    {
        std::vector<double> total_partitions_p;
        total_partitions.push_back(total_partitions_p);
    }

    gpu_error_check(cudaSetDevice(0));
    for(const auto& block: d_phi_block)
    {
        const auto& key = block.first;
        int p                 = std::get<0>(key);
        std::string key_left  = std::get<1>(key);
        std::string key_right = std::get<2>(key);

        int n_segment_right = propagator_analyzer->get_computation_block(key).n_segment_right;
        int n_segment_left  = propagator_analyzer->get_computation_block(key).n_segment_left;
        int n_repeated      = propagator_analyzer->get_computation_block(key).n_repeated;
        int n_propagators   = propagator_analyzer->get_computation_block(key).v_u.size();

        #ifndef NDEBUG
        std::cout<< p << ", " << key_left << ", " << key_right << ": " << n_segment_left << ", " << n_segment_right << ", " << n_propagators << ", " << propagator_analyzer->get_computation_block(key).n_repeated << std::endl;
        #endif

        for(int n=0;n<=n_segment_right;n++)
        {
            double total_partition = cb->inner_product_device(
                d_propagator[key_left][n_segment_left-n],
                d_propagator[key_right][n])*n_repeated/cb->get_volume();
            
            total_partition /= n_propagators;
            total_partitions[p].push_back(total_partition);

            #ifndef NDEBUG
            std::cout<< p << ", " << n << ": " << total_partition << std::endl;
            #endif
        }
    }

    // Find minimum and maximum of total_partitions
    std::cout<< "Polymer id: maximum,  minimum, and difference of total partitions" << std::endl;
    for(size_t p=0;p<total_partitions.size();p++)
    {
        double max_partition = -1e20;
        double min_partition =  1e20;
        for(size_t n=0;n<total_partitions[p].size();n++)
        {
            if (total_partitions[p][n] > max_partition)
                max_partition = total_partitions[p][n];
            if (total_partitions[p][n] < min_partition)
                min_partition = total_partitions[p][n];
        }
        double diff_partition = std::abs(max_partition - min_partition);

        std::cout<< "\t" << p << ": " << max_partition << ", " << min_partition << ", " << diff_partition << std::endl;
        if (diff_partition > 1e-7)
            return false;
    }
    return true;
}