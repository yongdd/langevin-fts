#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <complex>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include "CudaPseudoContinuous.h"
#include "CudaComputationBox.h"
#include "SimpsonQuadrature.h"

CudaPseudoContinuous::CudaPseudoContinuous(
    ComputationBox *cb,
    PolymerChain *pc)
    : Pseudo(cb, pc)
{
    try{
        const int M = cb->get_n_grid();
        const int N_B = pc->get_n_block();
        const int N = pc->get_n_segment_total();
        const int M_COMPLEX = this->n_complex_grid;

        // Create FFT plan
        const int BATCH{2};
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
            n_grid[0] = cb->get_nx(1);
            n_grid[1] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 1)
        {
            n_grid[0] = cb->get_nx(2);
        }
        cufftPlanMany(&plan_for, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z,BATCH);
        cufftPlanMany(&plan_bak, NRANK, n_grid, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D,BATCH);

        // Memory allocation
        gpu_error_check(cudaMalloc((void**)&d_q_step1, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_step2, sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_k_q_in,   sizeof(ftsComplex)*2*M_COMPLEX));

        gpu_error_check(cudaMalloc((void**)&d_q_1, sizeof(double)*M*(N+1)));
        gpu_error_check(cudaMalloc((void**)&d_q_2, sizeof(double)*M*(N+1)));
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M*N_B));

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: pc->get_dict_bond_lengths()){
            std::string species = item.first;
            d_boltz_bond     [species] = nullptr;
            d_boltz_bond_half[species] = nullptr;
            d_exp_dw         [species] = nullptr;
            d_exp_dw_half    [species] = nullptr;

            gpu_error_check(cudaMalloc((void**)&d_exp_dw         [species], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_exp_dw_half    [species], sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond     [species], sizeof(double)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_boltz_bond_half[species], sizeof(double)*M_COMPLEX));
        }
        
        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CudaPseudoContinuous::~CudaPseudoContinuous()
{
    cufftDestroy(plan_for);
    cufftDestroy(plan_bak);

    cudaFree(d_q_step1);
    cudaFree(d_q_step2);
    cudaFree(d_k_q_in);

    cudaFree(d_q_1);
    cudaFree(d_q_2);
    cudaFree(d_phi);

    for(const auto& item: d_boltz_bond)
        cudaFree(item.second);
    for(const auto& item: d_boltz_bond_half)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw_half)
        cudaFree(item.second);
}

void CudaPseudoContinuous::update()
{
    try{
        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: pc->get_dict_bond_lengths()){
            std::string species = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), pc->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), pc->get_ds());
        
            gpu_error_check(cudaMemcpy(d_boltz_bond[species],      boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[species], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        }
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
std::vector<int> CudaPseudoContinuous::get_block_start()
{
    std::vector<int> seg_start;
    seg_start.push_back(0);
    int seg_start_temp = 0;
    for(int i=0; i<pc->get_n_block(); i++){
        seg_start_temp += pc->get_n_segment(i);
        seg_start.push_back(seg_start_temp);
    }
    return seg_start;
}
std::array<double,3> CudaPseudoContinuous::dq_dl()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int N    = pc->get_n_segment_total();
        const int N_B  = pc->get_n_block();
        const int M_COMPLEX = this->n_complex_grid;

        std::array<double,3> dq_dl;
        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        double *d_fourier_basis_x;
        double *d_fourier_basis_y;
        double *d_fourier_basis_z;
        double *d_q_in_2m, *d_q_multi, *d_stress_sum;

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());

        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_in_2m,         sizeof(double)*2*M));
        gpu_error_check(cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX));

        gpu_error_check(cudaMemcpy(d_fourier_basis_x, fourier_basis_x, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_y, fourier_basis_y, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_fourier_basis_z, fourier_basis_z, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));

        thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

        std::map<std::string, double>& dict_bond_lengths = pc->get_dict_bond_lengths();
        auto seg_start = get_block_start();

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;

        for(int b=0; b<N_B; b++)
        {
            std::vector<double> simpson_rule_coeff = SimpsonQuadrature::get_coeff(pc->get_n_segment(b));
            std::string species = pc->get_block_species(b);
            double bond_length_sq = dict_bond_lengths[species]*dict_bond_lengths[species];

            for(int n=seg_start[b]; n<=seg_start[b+1]; n++)
            {
                double s_coeff = simpson_rule_coeff[n-seg_start[b]];

                gpu_error_check(cudaMemcpy(&d_q_in_2m[0], &d_q_1[n*M],     sizeof(double)*M,cudaMemcpyDeviceToDevice));
                gpu_error_check(cudaMemcpy(&d_q_in_2m[M], &d_q_2[(N-n)*M], sizeof(double)*M,cudaMemcpyDeviceToDevice));
                cufftExecD2Z(plan_for, d_q_in_2m, d_k_q_in);

                multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, &d_k_q_in[0], &d_k_q_in[M_COMPLEX], M_COMPLEX);
                if ( DIM >= 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, bond_length_sq, M_COMPLEX);
                    dq_dl[0] += s_coeff*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_sq, M_COMPLEX);
                    dq_dl[1] += s_coeff*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    dq_dl[2] += s_coeff*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
            }
        }

        for(int d=0; d<3; d++)
            dq_dl[d] /= 3.0*cb->get_lx(d)*M*M/pc->get_ds()/cb->get_volume();

        cudaFree(d_fourier_basis_x);
        cudaFree(d_fourier_basis_y);
        cudaFree(d_fourier_basis_z);
        cudaFree(d_q_in_2m);
        cudaFree(d_q_multi);
        cudaFree(d_stress_sum);

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::calculate_phi_one_type(
    double *d_phi, const int N_START, const int N_END)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int N = pc->get_n_segment_total();
        auto simpson_rule_coeff = SimpsonQuadrature::get_coeff(N_END-N_START);

        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[M*N_START], &d_q_2[M*(N-N_START)], simpson_rule_coeff[0], M);
        for(int n=N_START+1; n<=N_END; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[M*n], &d_q_2[M*(N-n)], simpson_rule_coeff[n-N_START], M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CudaPseudoContinuous::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block,
    double *phi, double &single_partition)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M     = cb->get_n_grid();
        const int N = pc->get_n_segment_total();
        const int N_B = pc->get_n_block();
        const double ds = pc->get_ds();
        auto seg_start = get_block_start();

        for(int i=0; i<pc->get_n_block(); i++)
        {
            if( w_block.count(pc->get_block_species(i)) == 0)
                throw_with_line_number("\"" + pc->get_block_species(i) + "\" species is not in w_block.");
        }

        // exp_dw and exp_dw_half
        double exp_dw[M];
        double exp_dw_half[M];
        for(const auto& item: w_block)
        {
            std::string species = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
            { 
                exp_dw     [i] = exp(-w[i]*ds*0.5);
                exp_dw_half[i] = exp(-w[i]*ds*0.25);
            }
            gpu_error_check(cudaMemcpy(d_exp_dw[species],      exp_dw,      sizeof(double)*M,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_exp_dw_half[species], exp_dw_half, sizeof(double)*M,cudaMemcpyHostToDevice));
        }

        // initial conditions
        if (q_init.count("1") == 0)
        {
            double q_1_init[M];
            for(int i=0; i<M; i++)
                q_1_init[i] = 1.0;
            gpu_error_check(cudaMemcpy(&d_q_1[0], q_1_init, sizeof(double)*M,
                cudaMemcpyHostToDevice));
        }
        else
        {
            gpu_error_check(cudaMemcpy(&d_q_1[0], q_init["1"], sizeof(double)*M,
                cudaMemcpyHostToDevice));
        }

        if (q_init.count("2") == 0)
        {     
            double q_2_init[M];
            for(int i=0; i<M; i++)
                q_2_init[i] = 1.0;
            gpu_error_check(cudaMemcpy(&d_q_2[0], q_2_init, sizeof(double)*M,
                cudaMemcpyHostToDevice));
        }
        else
        {
            gpu_error_check(cudaMemcpy(&d_q_2[0], q_init["1"], sizeof(double)*M,
                    cudaMemcpyHostToDevice));
        }

        int b_curr_1, b_curr_2; // current block of q1/q2
        for(int n=0; n<N; n++)
        {
            // find currently working block 
            int b_ = 0;
            while(b_ < N && seg_start[b_] <= n)  /////////////////////////reminder to revisit here to check error
            {
                b_curr_1 = b_;
                b_++;
            }
            b_ = N_B-1;
            while(b_ >= 0 && N - seg_start[b_+1] <= n)
            {
                b_curr_2 = b_;
                b_--;
            }
            std::string species1 = pc->get_block_species(b_curr_1);
            std::string species2 = pc->get_block_species(b_curr_2);

            one_step(
                    &d_q_1[M*n], &d_q_1[M*(n+1)],
                    &d_q_2[M*n], &d_q_2[M*(n+1)],
                    d_boltz_bond[species1], d_boltz_bond_half[species1],
                    d_boltz_bond[species2], d_boltz_bond_half[species2],
                    d_exp_dw[species1], d_exp_dw_half[species1],
                    d_exp_dw[species2], d_exp_dw_half[species2]);
        }

        // calculates the total partition function
        //d_phi is used as a temporary array
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_phi[0], &d_q_1[M*N], &d_q_2[0], 0.5, M);//&d_q_1[M*N_A], &d_q_2[M*(N-N_A)], 0.5, M);
        single_partition = 2*((CudaComputationBox *)cb)->integral_gpu(&d_phi[0]);

        // segment concentration.
        for(int b=0; b<N_B; b++)
        {
            calculate_phi_one_type(&d_phi[b*M], seg_start[b], seg_start[b+1]);
        }

        // normalize the concentration
        for(int b=0; b<N_B; b++)
            lin_comb<<<N_BLOCKS, N_THREADS>>>(&d_phi[b*M], cb->get_volume()*pc->get_ds()/single_partition, &d_phi[b*M], 0.0, &d_phi[b*M], M);

        for(int b=0; b<N_B; b++)
            gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*N_B*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Advance two partial partition functions simultaneously using Richardson extrapolation.
// Note that cufft doesn't fully utilize GPU cores unless n_grid is sufficiently large.
// To increase GPU usage, FFT Batch is utilized.
void CudaPseudoContinuous::one_step(double *d_q_in1,        double *d_q_out1,
                                  double *d_q_in2,        double *d_q_out2,
                                  double *d_boltz_bond_1, double *d_boltz_bond_1_half,
                                  double *d_boltz_bond_2, double *d_boltz_bond_2_half,
                                  double *d_exp_dw_1,     double *d_exp_dw_1_half,
                                  double *d_exp_dw_2,     double *d_exp_dw_2_half)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        //-------------- step 1 ----------
        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[0], d_q_in1, d_exp_dw_1, 1.0, M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[M], d_q_in2, d_exp_dw_2, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step1, d_k_q_in);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[0],         d_boltz_bond_1, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[M_COMPLEX], d_boltz_bond_2, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_k_q_in, d_q_step1);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[0], &d_q_step1[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step1[M], &d_q_step1[M], d_exp_dw_2, 1.0/((double)M), M);

        //-------------- step 2 ----------
        // Evaluate e^(-w*ds/4) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[0], d_q_in1, d_exp_dw_1_half, 1.0, M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[M], d_q_in2, d_exp_dw_2_half, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_k_q_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[0],         d_boltz_bond_1_half, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[M_COMPLEX], d_boltz_bond_2_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_k_q_in, d_q_step2);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[0], &d_q_step2[0], d_exp_dw_1, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[M], &d_q_step2[M], d_exp_dw_2, 1.0/((double)M), M);
        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_k_q_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[0],         d_boltz_bond_1_half, M_COMPLEX);
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(&d_k_q_in[M_COMPLEX], d_boltz_bond_2_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_k_q_in, d_q_step2);

        // Evaluate e^(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[0], &d_q_step2[0], d_exp_dw_1_half, 1.0/((double)M), M);
        multi_real<<<N_BLOCKS, N_THREADS>>>(&d_q_step2[M], &d_q_step2[M], d_exp_dw_2_half, 1.0/((double)M), M);
        //-------------- step 3 ----------
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out1, 4.0/3.0, &d_q_step2[0], -1.0/3.0, &d_q_step1[0], M);
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out2, 4.0/3.0, &d_q_step2[M], -1.0/3.0, &d_q_step1[M], M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::get_partition(double *q_out, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        const int b = pc->get_array_idx(v,u);
        const int N = pc->get_n_segment(b);
        auto block_start = get_block_start();

        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        if (v < u)
        {
            gpu_error_check(cudaMemcpy(q_out, &d_q_1[(block_start[b] + n)*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
        }
        else
        {
            gpu_error_check(cudaMemcpy(q_out, &d_q_2[(N-block_start[b] + n)*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}