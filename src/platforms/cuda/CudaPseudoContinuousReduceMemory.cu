#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "CudaPseudoContinuousReduceMemory.h"
#include "CudaComputationBox.h"
#include "SimpsonQuadrature.h"

CudaPseudoContinuousReduceMemory::CudaPseudoContinuousReduceMemory(
    ComputationBox *cb,
    Mixture *mx)
    : Pseudo(cb, mx)
{
    try{
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        // allocate memory for partition functions
        if( mx->get_unique_branches().size() == 0)
            throw_with_line_number("There is no unique branch. Add polymers first.");
        for(const auto& item: mx->get_unique_branches())
        {
            std::string dep = item.first;
            int max_n_segment = item.second.max_n_segment;

            // allocate pinned memory for device overlapping
            cudaMallocHost((void**)&unique_partition[dep], sizeof(double)*(max_n_segment+1)*M);

            #ifndef NDEBUG
            unique_partition_finished[dep] = new bool[max_n_segment+1];
            for(int i=0; i<=max_n_segment;i++)
                unique_partition_finished[dep][i] = false;
            #endif
        }

        // allocate memory for concentrations
        if( mx->get_unique_blocks().size() == 0)
            throw_with_line_number("There is no unique block. Add polymers first.");
        for(const auto& item: mx->get_unique_blocks())
        {
            unique_phi[item.first] = new double[M];
        }

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: mx->get_bond_lengths()){
            std::string monomer_type = item.first;
            d_boltz_bond     [monomer_type] = nullptr;
            d_boltz_bond_half[monomer_type] = nullptr;
            d_exp_dw         [monomer_type] = nullptr;
            d_exp_dw_half    [monomer_type] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_exp_dw         [monomer_type], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw_half    [monomer_type], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [monomer_type], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[monomer_type], sizeof(double)*M_COMPLEX));
        }

        // total partition functions for each polymer
        single_partitions = new double[mx->get_n_polymers()];

        // create FFT plan
        const int NRANK{cb->get_dim()};
        int n_grid[NRANK];

        if(cb->get_dim() == 3)
        {
            n_grid[0] = cb->get_nx(0);
            n_grid[1] = cb->get_nx(1);
            n_grid[2] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 2)
        {
            n_grid[0] = cb->get_nx(0);
            n_grid[1] = cb->get_nx(1);
        }
        else if(cb->get_dim() == 1)
        {
            n_grid[0] = cb->get_nx(0);
        }

        // cufft plans using one batches for propagators 
        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,1);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,1);

        // cufft plan using two batches for stress computation
        cufftPlanMany(&plan_for_two, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,2);

        // three streams for overlapping kernel execution and data transfers 
        // streams[0] : data transfers
        // streams[1] : compute_statistics() using single batched cufft
        // streams[2] : compute_stress() using double batched cufft
        const int NUM_STREAMS = 3;
        streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; i++)
            cudaStreamCreate(&streams[i]);
        cufftSetStream(plan_for, streams[1]);
        cufftSetStream(plan_bak, streams[1]); 
        cufftSetStream(plan_for_two, streams[2]);

        // allocate memory for pseudo-spectral: one_step()
        d_q = new double*[2]; // one for prev, the other for next
        gpu_error_check(cudaMalloc((void**)&d_q[0], sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q[1], sizeof(double)*M));

        gpu_error_check(cudaMalloc((void**)&d_q_step1, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_step2, sizeof(double)*M));

        gpu_error_check(cudaMalloc((void**)&d_qk_in,  sizeof(ftsComplex)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_unique_partition_sub_dep, sizeof(double)*M));

        // allocate memory for stress calculation: compute_stress()
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));        
        gpu_error_check(cudaMalloc((void**)&d_two_qk_in, sizeof(ftsComplex)*2*M_COMPLEX));

        const int NUM_BATCHES_FOR_STRESS = 2;
        d_q_two_partition = new double*[NUM_BATCHES_FOR_STRESS];
        for (int i=0; i<NUM_BATCHES_FOR_STRESS; i++)
            gpu_error_check(cudaMalloc((void**)&d_q_two_partition[i], sizeof(double)*2*M));

        gpu_error_check(cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX));

        update_bond_function();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoContinuousReduceMemory::~CudaPseudoContinuousReduceMemory()
{
    const int NUM_STREAMS = 3;
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    free(streams);

    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);
    cufftDestroy(plan_for_two);

    delete[] single_partitions;

    for(const auto& item: d_boltz_bond)
        cudaFree(item.second);
    for(const auto& item: d_boltz_bond_half)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw_half)
        cudaFree(item.second);

    for(const auto& item: unique_partition)
        cudaFreeHost(item.second);
    for(const auto& item: unique_phi)
        delete[] item.second;

    #ifndef NDEBUG
    for(const auto& item: unique_partition_finished)
        delete[] item.second;
    #endif

    // for pseudo-spectral: one_step()
    cudaFree(d_q[0]);
    cudaFree(d_q[1]);
    delete[] d_q;

    cudaFree(d_q_step1);
    cudaFree(d_q_step2);
    cudaFree(d_qk_in);
    cudaFree(d_unique_partition_sub_dep);

    // for stress calculation: compute_stress()
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    cudaFree(d_two_qk_in);

    const int NUM_BATCHES_FOR_STRESS = 2;
    for (int i=0; i<NUM_BATCHES_FOR_STRESS; i++)
        cudaFree(d_q_two_partition[i]);
    delete[] d_q_two_partition;

    cudaFree(d_q_multi);
    cudaFree(d_stress_sum);
}

void CudaPseudoContinuousReduceMemory::update_bond_function()
{
    try{
        // for pseudo-spectral: one_step()
        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: mx->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), mx->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), mx->get_ds());
        
            gpu_error_check(cudaMemcpy(d_boltz_bond[monomer_type],      boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[monomer_type], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        }

        // for stress calculation: compute_stress()
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];
        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        gpu_error_check(cudaMemcpy(d_fourier_basis_x, fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_y, fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_z, fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
void CudaPseudoContinuousReduceMemory::compute_statistics(
    std::map<std::string, double*> w_input,
    std::map<std::string, double*> q_init)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_unique_branches())
        {
            if( w_input.find(item.second.monomer_type) == w_input.end())
                throw_with_line_number("monomer_type \"" + item.second.monomer_type + "\" is not in w_input.");
        }

        for(const auto& item: w_input)
        {
            if( d_exp_dw.find(item.first) == d_exp_dw.end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");     
        }

        // exp_dw and exp_dw_half
        double exp_dw[M];
        double exp_dw_half[M];
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
            { 
                exp_dw     [i] = exp(-w[i]*ds*0.5);
                exp_dw_half[i] = exp(-w[i]*ds*0.25);
            }
            gpu_error_check(cudaMemcpy(d_exp_dw     [monomer_type], exp_dw,      sizeof(double)*M,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_exp_dw_half[monomer_type], exp_dw_half, sizeof(double)*M,cudaMemcpyHostToDevice));
        }

        double q_uniform[M];
        for(int i=0; i<M; i++)
            q_uniform[i] = 1.0;

        // for each unique branch
        for (auto& item: mx->get_unique_branches())
        {
            auto& key = item.first;
            auto& deps = item.second.deps;
            int n_segment = item.second.max_n_segment;
            auto monomer_type = item.second.monomer_type;

            // check key
            #ifndef NDEBUG
            if (unique_partition.find(key) == unique_partition.end())
                throw_with_line_number("Could not find key '" + key + "'. ");
            #endif
            double *_unique_partition = unique_partition[key];

            // if it is leaf node
            if(deps.size() == 0) 
            {
                // q_init
                if (key[0] == '{')
                {
                    std::string g = Mixture::get_q_input_idx_from_key(key);
                    if (q_init.find(g) == q_init.end())
                        throw_with_line_number( "Could not find q_init[\"" + g + "\"].");
                    gpu_error_check(cudaMemcpy(d_q[0], q_init[g], sizeof(double)*M, cudaMemcpyHostToDevice));
                }
                else
                {
                    gpu_error_check(cudaMemcpy(d_q[0], q_uniform, sizeof(double)*M, cudaMemcpyHostToDevice));
                }

                #ifndef NDEBUG
                unique_partition_finished[key][0] = true;
                #endif
            }
            // if it is not leaf node
            else if (deps.size() > 0) 
            {
                // if it is superposed
                if (key[0] == '[')
                {
                    // initialize to zero
                    gpu_error_check(cudaMemset(d_q[0], 0, sizeof(double)*M));

                    for(size_t d=0; d<deps.size(); d++)
                    {
                        std::string sub_dep = std::get<0>(deps[d]);
                        int sub_n_segment   = std::get<1>(deps[d]);
                        int sub_n_repeated  = std::get<2>(deps[d]);

                        // check sub key
                        #ifndef NDEBUG
                        if (unique_partition.find(sub_dep) == unique_partition.end())
                            throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                        if (!unique_partition_finished[sub_dep][sub_n_segment])
                            throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                        #endif

                        gpu_error_check(cudaMemcpy(d_unique_partition_sub_dep, &unique_partition[sub_dep][sub_n_segment*M], sizeof(double)*M, cudaMemcpyHostToDevice));
                        lin_comb<<<N_BLOCKS, N_THREADS>>>(
                                d_q[0], 1.0, d_q[0],
                                sub_n_repeated, d_unique_partition_sub_dep, M);
                    }

                    #ifndef NDEBUG
                    unique_partition_finished[key][0] = true;
                    #endif
                }
                else
                { 
                    // initialize to one
                    gpu_error_check(cudaMemcpy(d_q[0], q_uniform,
                        sizeof(double)*M, cudaMemcpyHostToDevice));

                    for(size_t d=0; d<deps.size(); d++)
                    {
                        std::string sub_dep = std::get<0>(deps[d]);
                        int sub_n_segment   = std::get<1>(deps[d]);

                        // check sub key
                        #ifndef NDEBUG
                        if (unique_partition.find(sub_dep) == unique_partition.end())
                            throw_with_line_number("Could not find sub key '" + sub_dep + "'. ");
                        if (!unique_partition_finished[sub_dep][sub_n_segment])
                            throw_with_line_number("Could not compute '" + key +  "', since '"+ sub_dep + std::to_string(sub_n_segment) + "' is not prepared.");
                        #endif

                        gpu_error_check(cudaMemcpy(d_unique_partition_sub_dep, &unique_partition[sub_dep][sub_n_segment*M], sizeof(double)*M, cudaMemcpyHostToDevice));
                        multi_real<<<N_BLOCKS, N_THREADS>>>(
                            d_q[0], d_q[0], d_unique_partition_sub_dep, 1.0, M);

                    }
                    
                    #ifndef NDEBUG
                    unique_partition_finished[key][0] = true;
                    #endif
                }
            }
            cudaDeviceSynchronize();

            // if there is no segment to be computed
            if (n_segment == 0)
            {
                gpu_error_check(cudaMemcpy(&_unique_partition[0], d_q[0], sizeof(double)*M, cudaMemcpyDeviceToHost));
                continue;
            }

            // apply the propagator successively
            int prev, next, swap;
            prev = 0;
            next = 1;

            for(int n=1; n<=n_segment; n++)
            {
                #ifndef NDEBUG
                if (!unique_partition_finished[key][n-1])
                    throw_with_line_number("unfinished, key: " + key + ", " + std::to_string(n-1));
                #endif

                // STREAM 0: copy memory from device to host
                gpu_error_check(cudaMemcpyAsync(&_unique_partition[(n-1)*M], d_q[prev], sizeof(double)*M,
                    cudaMemcpyDeviceToHost, streams[0]));

                // STREAM 1: calculate partial partition function
                one_step(
                    d_q[prev],
                    d_q[next],
                    d_boltz_bond[monomer_type],
                    d_boltz_bond_half[monomer_type],
                    d_exp_dw[monomer_type],
                    d_exp_dw_half[monomer_type]);

                swap = next;
                next = prev;
                prev = swap;
                cudaDeviceSynchronize();

                #ifndef NDEBUG
                unique_partition_finished[key][n] = true;
                #endif
            }
            gpu_error_check(cudaMemcpy(&_unique_partition[(n_segment)*M], d_q[prev], sizeof(double)*M,
                cudaMemcpyDeviceToHost));
        }

        // compute total partition function of each distinct polymers
        int current_p = 0;
        for(const auto& block: unique_phi)
        {
            int p                = std::get<0>(block.first);
            std::string dep_v    = std::get<1>(block.first);
            std::string dep_u    = std::get<2>(block.first);

            // already computed
            if (p != current_p)
                continue;

            int n_superposed;
            // int n_segment_allocated = mx->get_unique_block(block.first).n_segment_allocated;
            int n_segment_offset    = mx->get_unique_block(block.first).n_segment_offset;
            int n_segment_original  = mx->get_unique_block(block.first).n_segment_original;

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_superposed = 1;
            else
                n_superposed = mx->get_unique_block(block.first).v_u.size();

            // check keys
            #ifndef NDEBUG
            if (unique_partition.find(dep_v) == unique_partition.end())
                std::cout << "Could not find dep_v key'" + dep_v + "'. " << std::endl;
            if (unique_partition.find(dep_u) == unique_partition.end())
                std::cout << "Could not find dep_u key'" + dep_u + "'. " << std::endl;
            #endif

            single_partitions[p]= cb->inner_product(
                &unique_partition[dep_v][(n_segment_original-n_segment_offset)*M], // q
                &unique_partition[dep_u][0])/n_superposed/cb->get_volume();        // q^dagger

            // std::cout << p << ", " << single_partitions[p] << std::endl;
            // std::cout << p << ", "<< dep_v << ", "<< dep_u << ", "<< single_partitions[p] << std::endl;
            // std::cout << p <<", "<< n_segment <<", "<< n_segment_offset <<", "<< single_partitions[p] << std::endl;
            current_p++;
        }

        // calculate segment concentrations
        for(size_t b=0; b<unique_phi.size();b++)
        {
            auto block = unique_phi.begin();
            advance(block, b);
            const auto& key = block->first;

            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            int n_repeated;
            int n_segment_allocated = mx->get_unique_block(key).n_segment_allocated;
            int n_segment_offset    = mx->get_unique_block(key).n_segment_offset;
            int n_segment_original  = mx->get_unique_block(key).n_segment_original;

            // if there is no segment
            if(n_segment_allocated == 0)
            {
                for(int i=0; i<M;i++)
                    block->second[i] = 0.0;
                continue;
            }

            // contains no '['
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(key).v_u.size();
            else
                n_repeated = 1;

            // check keys
            #ifndef NDEBUG
            if (unique_partition.find(dep_v) == unique_partition.end())
                std::cout << "Could not find dep_v key'" + dep_v + "'. " << std::endl;
            if (unique_partition.find(dep_u) == unique_partition.end())
                std::cout << "Could not find dep_u key'" + dep_u + "'. " << std::endl;
            #endif

            // calculate phi of one block (possibly multiple blocks when using superposition)
            calculate_phi_one_block(
                block->second,             // phi
                unique_partition[dep_v],  // dependency v
                unique_partition[dep_u],  // dependency u
                n_segment_allocated,
                n_segment_offset,
                n_segment_original);

            // normalize concentration
            PolymerChain& pc = mx->get_polymer(p);
            double norm = mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p]*n_repeated;
            for(int i=0; i<M; i++)
                block->second[i] *= norm;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Advance partial partition function using Richardson extrapolation.
void CudaPseudoContinuousReduceMemory::one_step(
    double *d_q_in, double *d_q_out,
    double *d_boltz_bond, double *d_boltz_bond_half,
    double *d_exp_dw, double *d_exp_dw_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_q_step1, d_q_in, d_exp_dw, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step1, d_qk_in);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_qk_in, d_boltz_bond, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_qk_in, d_q_step1);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_q_step1, d_q_step1, d_exp_dw, 1.0/((double)M), M);

        //-------------- step 2 ----------
        // Evaluate e^(-w*ds/4) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_q_step2, d_q_in, d_exp_dw_half, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_qk_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_qk_in, d_boltz_bond_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_qk_in, d_q_step2);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_q_step2, d_q_step2, d_exp_dw, 1.0/((double)M), M);
        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_qk_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_qk_in, d_boltz_bond_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_qk_in, d_q_step2);

        // Evaluate e^(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_q_step2, d_q_step2, d_exp_dw_half, 1.0/((double)M), M);
        //-------------- step 3 ----------
        lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[1]>>>(d_q_out, 4.0/3.0, d_q_step2, -1.0/3.0, d_q_step1, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuousReduceMemory::calculate_phi_one_block(
    double *phi, double *q_1, double *q_2, const int N, const int N_OFFSET, const int N_ORIGINAL)
{

    try
    {
        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonQuadrature::get_coeff(N);

        // Compute segment concentration
        for(int i=0; i<M; i++)
            phi[i] = simpson_rule_coeff[0]*q_1[i+(N_ORIGINAL-N_OFFSET)*M]*q_2[i];
        for(int n=1; n<=N; n++)
        {
            for(int i=0; i<M; i++)
                phi[i] += simpson_rule_coeff[n]*q_1[i+(N_ORIGINAL-N_OFFSET-n)*M]*q_2[i+n*M];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaPseudoContinuousReduceMemory::get_total_partition(int polymer)
{
    try
    {
        return single_partitions[polymer];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuousReduceMemory::get_monomer_concentration(std::string monomer_type, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        // initialize array
        for(int i=0; i<M; i++)
            phi[i] = 0.0;

        // for each block
        for(const auto& block: unique_phi)
        {
            std::string dep_v = std::get<1>(block.first);
            int n_segment_allocated = mx->get_unique_block(block.first).n_segment_allocated;
            if (Mixture::get_monomer_type_from_key(dep_v) == monomer_type && n_segment_allocated != 0)
            {
                for(int i=0; i<M; i++)
                    phi[i] += block.second[i]; 
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuousReduceMemory::get_polymer_concentration(int p, double *phi)
{
    try
    {
        const int M = cb->get_n_grid();
        const int P = mx->get_n_polymers();

        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (mx->is_using_superposition())
            throw_with_line_number("Disable 'superposition' option to invoke 'get_polymer_concentration'.");

        PolymerChain& pc = mx->get_polymer(p);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(size_t b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
            if (dep_v < dep_u)
                dep_v.swap(dep_u);

            double* _unique_phi = unique_phi[std::make_tuple(p, dep_v, dep_u)];
            for(int i=0; i<M; i++)
                phi[i+b*M] = _unique_phi[i]; 
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<double> CudaPseudoContinuousReduceMemory::compute_stress()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        auto bond_lengths = mx->get_bond_lengths();
        std::vector<double> stress(cb->get_dim());
        std::map<std::tuple<int, std::string, std::string>, std::array<double,3>> unique_dq_dl;
        thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

        // reset stress map
        for(const auto& item: unique_phi)
        {
            for(int d=0; d<3; d++)
                unique_dq_dl[item.first][d] = 0.0;
        }

        // compute stress for unique block
        for(const auto& block: unique_phi)
        {
            const auto& key      = block.first;
            int p                = std::get<0>(key);
            std::string dep_v    = std::get<1>(key);
            std::string dep_u    = std::get<2>(key);

            const int N           = mx->get_unique_block(block.first).n_segment_allocated;
            const int N_OFFSET    = mx->get_unique_block(block.first).n_segment_offset;
            const int N_ORIGINAL  = mx->get_unique_block(block.first).n_segment_original;
            std::string monomer_type = mx->get_unique_block(key).monomer_type;

            // if there is no segment
            if(N == 0)
                continue;

            // contains no '['
            int n_repeated;
            if (dep_u.find('[') == std::string::npos)
                n_repeated = mx->get_unique_block(block.first).v_u.size();
            else
                n_repeated = 1;

            std::vector<double> s_coeff = SimpsonQuadrature::get_coeff(N);
            double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            double* q_1 = unique_partition[dep_v];    // dependency v
            double* q_2 = unique_partition[dep_u];    // dependency u

            std::array<double,3> _unique_dq_dl = unique_dq_dl[key];

            int prev, next, swap;
            prev = 0;
            next = 1;

            // copy memory from device to host
            gpu_error_check(cudaMemcpy(&d_q_two_partition[prev][0], &q_1[(N_ORIGINAL-N_OFFSET)*M], sizeof(double)*M,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(&d_q_two_partition[prev][M], &q_2[0],                       sizeof(double)*M,cudaMemcpyHostToDevice));

            // compute
            for(int n=0; n<=N; n++)
            {
                // STREAM 0: copy memory from host to device
                if (n < N)
                {
                    gpu_error_check(cudaMemcpyAsync(&d_q_two_partition[next][0], &q_1[(N_ORIGINAL-N_OFFSET-(n+1))*M], sizeof(double)*M,cudaMemcpyHostToDevice, streams[0]));
                    gpu_error_check(cudaMemcpyAsync(&d_q_two_partition[next][M], &q_2[(n+1)*M],                       sizeof(double)*M,cudaMemcpyHostToDevice, streams[0]));
                }

                // STREAM 2: execute a Forward FFT
                cufftExecD2Z(plan_for_two, d_q_two_partition[prev], d_two_qk_in);

                // multiplay two partial partition functions in the fourier spaces
                multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[2]>>>(d_q_multi, &d_two_qk_in[0], &d_two_qk_in[M_COMPLEX], M_COMPLEX);

                if ( DIM == 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS, 0, streams[2]>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[0] += s_coeff[n]*thrust::reduce(thrust::cuda::par.on(streams[2]), temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS, 0, streams[2]>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[1] += s_coeff[n]*thrust::reduce(thrust::cuda::par.on(streams[2]), temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS, 0, streams[2]>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[2] += s_coeff[n]*thrust::reduce(thrust::cuda::par.on(streams[2]), temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                }
                if ( DIM == 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS, 0, streams[2]>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[0] += s_coeff[n]*thrust::reduce(thrust::cuda::par.on(streams[2]), temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;

                    multi_real<<<N_BLOCKS, N_THREADS, 0, streams[2]>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[1] += s_coeff[n]*thrust::reduce(thrust::cuda::par.on(streams[2]), temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }
                if ( DIM == 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS, 0, streams[2]>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    _unique_dq_dl[0] += s_coeff[n]*thrust::reduce(thrust::cuda::par.on(streams[2]), temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX)*n_repeated;
                }

                swap = next;
                next = prev;
                prev = swap;
                cudaDeviceSynchronize();
            }
            unique_dq_dl[key] = _unique_dq_dl;
        }

        // compute total stress
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] = 0.0;
        for(const auto& block: unique_phi)
        {
            const auto& key   = block.first;
            int p             = std::get<0>(key);
            std::string dep_v = std::get<1>(key);
            std::string dep_u = std::get<2>(key);
            PolymerChain& pc  = mx->get_polymer(p);

            for(int d=0; d<cb->get_dim(); d++)
                stress[d] += unique_dq_dl[key][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
        }
        for(int d=0; d<cb->get_dim(); d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/mx->get_ds();
            
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuousReduceMemory::get_partial_partition(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        PolymerChain& pc = mx->get_polymer(polymer);
        std::string dep = pc.get_dep(v,u);

        if (mx->get_unique_branches().find(dep) == mx->get_unique_branches().end())
            throw_with_line_number("Could not find the branches '" + dep + "'. Disable 'superposition' option to obtain partial partition functions.");

        const int N = mx->get_unique_branches()[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        double* _partition = unique_partition[dep];
        for(int i=0; i<M; i++)
            q_out[i] = _partition[n*M+i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}