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
            d_unique_partition[dep] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_unique_partition[dep], sizeof(double)*M*(max_n_segment+1)));
        }

        // allocate memory for concentrations
        if( mx->get_unique_blocks().size() == 0)
            throw_with_line_number("There is no unique block. Add polymers first.");
        for(const auto& item: mx->get_unique_blocks())
        {
            d_unique_phi[item.first] = nullptr;
            gpu_error_check(cudaMalloc((void**)&d_unique_phi[item.first], sizeof(double)*M));
        }

        // create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: mx->get_bond_lengths()){
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

        // total partition functions for each polymer
        single_partitions = new double[mx->get_n_polymers()];

        // create FFT plan
        const int BATCH{1};
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

        // allocate memory for get_concentration
        gpu_error_check(cudaMalloc((void**)&d_phi, sizeof(double)*M));

        // allocate memory for pseudo-spectral: one_step()
        gpu_error_check(cudaMalloc((void**)&d_q_step1, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_q_step2, sizeof(double)*M));
        gpu_error_check(cudaMalloc((void**)&d_qk_in,  sizeof(ftsComplex)*M_COMPLEX));
        
        // allocate memory for stress calculation: compute_stress()
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_x, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_y, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_fourier_basis_z, sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_qk_1,        sizeof(ftsComplex)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_qk_2,        sizeof(ftsComplex)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_q_multi,         sizeof(double)*M_COMPLEX));
        gpu_error_check(cudaMalloc((void**)&d_stress_sum,      sizeof(double)*M_COMPLEX));

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

    delete[] single_partitions;

    for(const auto& item: d_boltz_bond)
        cudaFree(item.second);
    for(const auto& item: d_boltz_bond_half)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw)
        cudaFree(item.second);
    for(const auto& item: d_exp_dw_half)
        cudaFree(item.second);
    for(const auto& item: d_unique_partition)
        cudaFree(item.second);
    for(const auto& item: d_unique_phi)
        cudaFree(item.second);

    // for get_concentration
    cudaFree(d_phi);

    // for pseudo-spectral: one_step()
    cudaFree(d_q_step1);
    cudaFree(d_q_step2);
    cudaFree(d_qk_in);

    // for stress calculation: compute_stress()
    cudaFree(d_fourier_basis_x);
    cudaFree(d_fourier_basis_y);
    cudaFree(d_fourier_basis_z);
    cudaFree(d_qk_1);
    cudaFree(d_qk_2);
    cudaFree(d_q_multi);
    cudaFree(d_stress_sum);
}

void CudaPseudoContinuous::update()
{
    try{
        // for pseudo-spectral: one_step()
        const int M_COMPLEX = this->n_complex_grid;
        double boltz_bond[M_COMPLEX], boltz_bond_half[M_COMPLEX];

        for(const auto& item: mx->get_bond_lengths())
        {
            std::string species = item.first;
            double bond_length_sq = item.second*item.second;
            get_boltz_bond(boltz_bond     , bond_length_sq,   cb->get_nx(), cb->get_dx(), mx->get_ds());
            get_boltz_bond(boltz_bond_half, bond_length_sq/2, cb->get_nx(), cb->get_dx(), mx->get_ds());
        
            gpu_error_check(cudaMemcpy(d_boltz_bond[species],      boltz_bond,      sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_boltz_bond_half[species], boltz_bond_half, sizeof(double)*M_COMPLEX,cudaMemcpyHostToDevice));
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
void CudaPseudoContinuous::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const double ds = mx->get_ds();

        for(const auto& item: mx->get_unique_branches())
        {
            if( w_block.count(item.second.species) == 0)
                throw_with_line_number("\"" + item.second.species + "\" species is not in w_block.");
        }

        if( q_init.size() > 0)
            throw_with_line_number("Currently, \'q_init\' is not supported.");

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
            gpu_error_check(cudaMemcpy(d_exp_dw     [species], exp_dw,      sizeof(double)*M,cudaMemcpyHostToDevice));
            gpu_error_check(cudaMemcpy(d_exp_dw_half[species], exp_dw_half, sizeof(double)*M,cudaMemcpyHostToDevice));
        }

        double q_uniform[M];
        for(int i=0; i<M; i++)
            q_uniform[i] = 1.0;
        for(const auto& item: mx->get_unique_branches())
        {
            auto& key = item.first;
            // calculate one block end
            if (item.second.deps.size() > 0) // if it is not leaf node
            {
                gpu_error_check(cudaMemcpy(d_unique_partition[key], q_uniform,
                    sizeof(double)*M, cudaMemcpyHostToDevice));

                for(int p=0; p<item.second.deps.size(); p++)
                {
                    std::string sub_dep = item.second.deps[p].first;
                    int sub_n_segment   = item.second.deps[p].second;
                    multi_real<<<N_BLOCKS, N_THREADS>>>(
                        d_unique_partition[key], d_unique_partition[key],
                        &d_unique_partition[sub_dep][sub_n_segment*M], 1.0, M);
                }
            }
            else // if it is leaf node
            {
                gpu_error_check(cudaMemcpy(d_unique_partition[key], q_uniform,
                    sizeof(double)*M, cudaMemcpyHostToDevice)); //* q_init
            }

            // apply the propagator successively
            for(int n=1; n<=item.second.max_n_segment; n++)
            {
                one_step(&d_unique_partition[key][(n-1)*M],
                         &d_unique_partition[key][n*M],
                         d_boltz_bond[item.second.species],
                         d_boltz_bond_half[item.second.species],
                         d_exp_dw[item.second.species],
                         d_exp_dw_half[item.second.species]);
            }
        }

        // calculate segment concentrations
        for(const auto& item: mx->get_unique_blocks())
        {
            auto& key = item.first;
            calculate_phi_one_type(
                d_unique_phi[key],                     // phi
                d_unique_partition[std::get<0>(key)],  // dependency v
                d_unique_partition[std::get<1>(key)],  // dependency u
                std::get<2>(key));                      // n_segment
        }

        // for each distinct polymers 
        for(int p=0; p<mx->get_n_polymers(); p++)
        {
            PolymerChain& pc = mx->get_polymer(p);
            std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

            // calculate the single chain partition function at block 0
            std::string dep_v = pc.get_dep(blocks[0].v, blocks[0].u);
            std::string dep_u = pc.get_dep(blocks[0].u, blocks[0].v);
            int n_segment = blocks[0].n_segment;
            single_partitions[p] = ((CudaComputationBox *)cb)->inner_product_gpu(
                &d_unique_partition[dep_v][n_segment*M],  // q
                &d_unique_partition[dep_u][0]);           // q^dagger
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Advance partial partition function using Richardson extrapolation.
void CudaPseudoContinuous::one_step(
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
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step1, d_q_in, d_exp_dw, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step1, d_qk_in);

        // Multiply e^(-k^2 ds/6) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in, d_boltz_bond, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_qk_in, d_q_step1);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step1, d_q_step1, d_exp_dw, 1.0/((double)M), M);

        //-------------- step 2 ----------
        // Evaluate e^(-w*ds/4) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step2, d_q_in, d_exp_dw_half, 1.0, M);

        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_qk_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in, d_boltz_bond_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_qk_in, d_q_step2);

        // Evaluate e^(-w*ds/2) in real space
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step2, d_q_step2, d_exp_dw, 1.0/((double)M), M);
        // Execute a Forw_ard FFT
        cufftExecD2Z(plan_for, d_q_step2, d_qk_in);

        // Multiply e^(-k^2 ds/12) in fourier space
        multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_qk_in, d_boltz_bond_half, M_COMPLEX);

        // Execute a backw_ard FFT
        cufftExecZ2D(plan_bak, d_qk_in, d_q_step2);

        // Evaluate e^(-w*ds/4) in real space.
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_q_step2, d_q_step2, d_exp_dw_half, 1.0/((double)M), M);
        //-------------- step 3 ----------
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_q_out, 4.0/3.0, d_q_step2, -1.0/3.0, d_q_step1, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::calculate_phi_one_type(
    double *d_phi, double *d_q_1, double *d_q_2, const int N)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        std::vector<double> simpson_rule_coeff = SimpsonQuadrature::get_coeff(N);

        // Compute segment concentration
        multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[0], &d_q_2[N*M], simpson_rule_coeff[0], M);
        for(int n=1; n<=N; n++)
        {
            add_multi_real<<<N_BLOCKS, N_THREADS>>>(d_phi, &d_q_1[n*M], &d_q_2[(N-n)*M], simpson_rule_coeff[n], M);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
double CudaPseudoContinuous::get_total_partition(int polymer)
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
void CudaPseudoContinuous::get_species_concentration(std::string species, double *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        // initialize to zero
        lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, 0.0, d_phi, M);

        // for each distinct polymers 
        for(int p=0; p<mx->get_n_polymers(); p++)
        {
            PolymerChain& pc = mx->get_polymer(p);
            std::vector<PolymerChainBlock>& blocks = pc.get_blocks();
            for(int b=0; b<blocks.size(); b++)
            {
                if (blocks[b].species == species)
                {
                    std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
                    std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
                    if (dep_v > dep_u)
                        dep_v.swap(dep_u);
                        
                    // normalize the concentration
                    double norm = cb->get_volume()*mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
                    lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 1.0, d_phi, norm, d_unique_phi[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)], M);
                }
            }
        }
        gpu_error_check(cudaMemcpy(phi, d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::get_polymer_concentration(int polymer, double *phi)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_n_grid();
        const int P = mx->get_n_polymers();

        if (polymer < 0 || polymer > P-1)
            throw_with_line_number("Index (" + std::to_string(polymer) + ") must be in range [0, " + std::to_string(P-1) + "]");

        PolymerChain& pc = mx->get_polymer(polymer);
        std::vector<PolymerChainBlock>& blocks = pc.get_blocks();

        for(int b=0; b<blocks.size(); b++)
        {
            std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
            std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
            if (dep_v > dep_u)
                dep_v.swap(dep_u);

            // copy normalized concentration
            double norm = cb->get_volume()*mx->get_ds()*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[polymer];
            lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, 0.0, d_phi, norm, d_unique_phi[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)], M);
            gpu_error_check(cudaMemcpy(&phi[b*M], d_phi, sizeof(double)*M, cudaMemcpyDeviceToHost));
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::array<double,3> CudaPseudoContinuous::compute_stress()
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

        std::map<std::string, double>& bond_lengths = mx->get_bond_lengths();
        std::array<double,3> stress;
        std::map<std::tuple<std::string, std::string, int>, std::array<double,3>> unique_dq_dl;
        thrust::device_ptr<double> temp_gpu_ptr(d_stress_sum);

        // compute stress for Unique key pairs
        for(const auto& item: mx->get_unique_blocks())
        {
            auto& key = item.first;
            std::string dep_v = std::get<0>(key);
            std::string dep_u = std::get<1>(key);
            const int N       = std::get<2>(key);
            std::string species = item.second.species;

            std::vector<double> s_coeff = SimpsonQuadrature::get_coeff(N);
            double bond_length_sq = bond_lengths[species]*bond_lengths[species];
            double* d_q_1 = d_unique_partition[dep_v];    // dependency v
            double* d_q_2 = d_unique_partition[dep_u];    // dependency u

            // reset
            for(int d=0; d<3; d++)
                unique_dq_dl[key][d] = 0.0;

            // compute
            for(int n=0; n<=N; n++)
            {
                cufftExecD2Z(plan_for, &d_q_1[n*M],     d_qk_1);
                cufftExecD2Z(plan_for, &d_q_2[(N-n)*M], d_qk_2);
                multi_complex_conjugate<<<N_BLOCKS, N_THREADS>>>(d_q_multi, d_qk_1, d_qk_2, M_COMPLEX);
                if ( DIM >= 3 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_x, bond_length_sq, M_COMPLEX);
                    unique_dq_dl[key][0] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 2 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_y, bond_length_sq, M_COMPLEX);
                    unique_dq_dl[key][1] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
                if ( DIM >= 1 )
                {
                    multi_real<<<N_BLOCKS, N_THREADS>>>(d_stress_sum, d_q_multi, d_fourier_basis_z, bond_length_sq, M_COMPLEX);
                    unique_dq_dl[key][2] += s_coeff[n]*thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + M_COMPLEX);
                }
            }
        }

        // compute total stress
        for(int d=0; d<3; d++)
            stress[d] = 0.0;
        for(int p=0; p < mx->get_n_polymers(); p++)
        {
            PolymerChain& pc = mx->get_polymer(p);
            std::vector<PolymerChainBlock>& blocks = pc.get_blocks();
            for(int b=0; b<blocks.size(); b++)
            {
                std::string dep_v = pc.get_dep(blocks[b].v, blocks[b].u);
                std::string dep_u = pc.get_dep(blocks[b].u, blocks[b].v);
                if (dep_v > dep_u)
                    dep_v.swap(dep_u);
                for(int d=0; d<3; d++)
                    stress[d] += unique_dq_dl[std::make_tuple(dep_v, dep_u, blocks[b].n_segment)][d]*pc.get_volume_fraction()/pc.get_alpha()/single_partitions[p];
            }
        }
        for(int d=0; d<3; d++)
            stress[d] /= -3.0*cb->get_lx(d)*M*M/mx->get_ds()/cb->get_volume();
            
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CudaPseudoContinuous::get_partial_partition(double *q_out, int polymer, int v, int u, int n)
{
    // This method should be invoked after invoking compute_statistics()

    // Get partial partition functions
    // This is made for debugging and testing
    try
    {
        const int M = cb->get_n_grid();
        PolymerChain& pc = mx->get_polymer(polymer);
        std::string dep = pc.get_dep(v,u);
        const int N = mx->get_unique_branches()[dep].max_n_segment;
        if (n < 0 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [0, " + std::to_string(N) + "]");

        double* partition = d_unique_partition[dep];
        gpu_error_check(cudaMemcpy(q_out, &partition[n*M], sizeof(double)*M,cudaMemcpyDeviceToHost));
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}