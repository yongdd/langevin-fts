#include <cmath>
#include "CpuPseudoLinearDiscrete.h"

CpuPseudoLinearDiscrete::CpuPseudoLinearDiscrete(
    ComputationBox *cb,
    PolymerChain *pc, FFT *fft)
    : Pseudo(cb, pc)
{
    try
    {
        const int M = cb->get_n_grid();
        const int N_B = pc->get_n_block();
        const int N = pc->get_n_segment_total();

        this->fft = fft;

        // create boltz_bond, boltz_bond_half, and exp_dw
        for(const auto& item: pc->get_dict_bond_lengths())
        {
            std::string species = item.first;
            exp_dw[species] = new double[M];
        }

        this->boltz_bond = new double*[N_B];
        if (N_B>1)
            this->boltz_bond_middle = new double*[N_B-1];

        this->boltz_bond[0]= new double[n_complex_grid];
        for (int b=0; b<N_B-1; b++)
        {
            this->boltz_bond[b+1]= new double[n_complex_grid];
            this->boltz_bond_middle[b] = new double[n_complex_grid];
        }

        // partial partition functions, q and q^dagger
        this->q_1 = new double[M*N];
        this->q_2 = new double[M*N];

        update();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuPseudoLinearDiscrete::~CpuPseudoLinearDiscrete()
{
    const int N_B = pc->get_n_block();
    delete fft;

    for(const auto& item: exp_dw)
        delete[] item.second;

    delete[] boltz_bond[0];
    for (int b=0; b<N_B-1; b++)
    {
        delete[] boltz_bond[b+1];
        delete[] boltz_bond_middle[b];
    }
    delete[] boltz_bond;
    if (N_B>1)
        delete[] boltz_bond_middle;
        
    delete[] q_1, q_2;
}
void CpuPseudoLinearDiscrete::update()
{
    try
    {
        const int N_B = pc->get_n_block();
        double bond_length_sq, bond_length_middle_sq;
        std::map<std::string, double>& dict_bond_lengths = pc->get_dict_bond_lengths();

        std::string species = pc->get_block_species(0);
        bond_length_sq = dict_bond_lengths[species]*dict_bond_lengths[species];
        get_boltz_bond(boltz_bond[0],  bond_length_sq,  cb->get_nx(), cb->get_dx(), pc->get_ds());

        for (int b=0; b<N_B-1; b++)
        {
            std::string species1 = pc->get_block_species(b);
            std::string species2 = pc->get_block_species(b+1);

            bond_length_sq = dict_bond_lengths[species1]*dict_bond_lengths[species1];
            bond_length_middle_sq =  0.5*dict_bond_lengths[species1]*dict_bond_lengths[species1]
                                   + 0.5*dict_bond_lengths[species2]*dict_bond_lengths[species2];

            get_boltz_bond(boltz_bond[b+1],       bond_length_sq,        cb->get_nx(), cb->get_dx(), pc->get_ds());
            get_boltz_bond(boltz_bond_middle[b],  bond_length_middle_sq,  cb->get_nx(), cb->get_dx(), pc->get_ds());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
std::vector<int> CpuPseudoLinearDiscrete::get_block_start()
{
    std::vector<int> seg_start;
    seg_start.push_back(0);
    int seg_start_temp = 0;
    for(int i=0; i<pc->get_n_block(); i++)
    {
        seg_start_temp += pc->get_n_segment(i);
        seg_start.push_back(seg_start_temp);
    }
    return seg_start;
}
std::array<double,3> CpuPseudoLinearDiscrete::dq_dl()
{
    // This method should be invoked after invoking compute_statistics().

    // To calculate stress, we multiply weighted fourier basis to q(k)*q^dagger(-k).
    // We only need the real part of stress calculation.

    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        const int N_B  = pc->get_n_block();
        const int N    = pc->get_n_segment_total();
        const int M_COMPLEX = this->n_complex_grid;
        double temp, bond_length_now, *boltz_bond_now;

        std::array<double,3> dq_dl;
        std::complex<double> k_q_1[M_COMPLEX];
        std::complex<double> k_q_2[M_COMPLEX];

        double fourier_basis_x[M_COMPLEX];
        double fourier_basis_y[M_COMPLEX];
        double fourier_basis_z[M_COMPLEX];

        get_weighted_fourier_basis(fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());

        std::map<std::string, double>& dict_bond_lengths = pc->get_dict_bond_lengths();
        auto seg_start = get_block_start();

        for(int i=0; i<3; i++)
            dq_dl[i] = 0.0;

        // compute stress within the blocks
        for(int b=0; b<N_B; b++)
        {
            std::string species = pc->get_block_species(b);
            double bond_length_sq = dict_bond_lengths[species]*dict_bond_lengths[species];
            boltz_bond_now = boltz_bond[b];

            for(int n=seg_start[b]+1; n<seg_start[b+1]; n++)
            {

                fft->forward(&q_1[(n-1)*M],k_q_1);
                fft->forward(&q_2[ n   *M],k_q_2);

                if ( DIM >= 3 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[0] += bond_length_sq*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
                }
                if ( DIM >= 2 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[1] += bond_length_sq*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
                }
                if ( DIM >= 1 )
                {
                    for(int i=0; i<M_COMPLEX; i++)
                        dq_dl[2] += bond_length_sq*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
                }
            }
        }

        // compute stress between the blocks
        for(int b=1; b<N_B; b++)
        {
            int n = seg_start[b];
            std::string species1 = pc->get_block_species(b-1);
            std::string species2 = pc->get_block_species(b);
            double bond_length_sq = 0.5*dict_bond_lengths[species1]*dict_bond_lengths[species1]
                + 0.5*dict_bond_lengths[species2]*dict_bond_lengths[species2];
            boltz_bond_now = boltz_bond_middle[b-1];

            fft->forward(&q_1[(n-1)*M],k_q_1);
            fft->forward(&q_2[ n   *M],k_q_2);

            if ( DIM >= 3 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[0] += bond_length_sq*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_x[i];
            }
            if ( DIM >= 2 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[1] += bond_length_sq*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_y[i];
            }
            if ( DIM >= 1 )
            {
                for(int i=0; i<M_COMPLEX; i++)
                    dq_dl[2] += bond_length_sq*boltz_bond_now[i]*(k_q_1[i]*std::conj(k_q_2[i])).real()*fourier_basis_z[i];
            }
        }
        
        for(int d=0; d<3; d++)
            dq_dl[d] /= 3.0*cb->get_lx(d)*M*M/pc->get_ds()/cb->get_volume();

        return dq_dl;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoLinearDiscrete::compute_statistics(
    std::map<std::string, double*> q_init,
    std::map<std::string, double*> w_block,
    double *phi, double &single_partition)
{
    try
    {
        const int M = cb->get_n_grid();
        const int N = pc->get_n_segment_total();
        const int N_B = pc->get_n_block();
        const double ds = pc->get_ds();
        auto seg_start = get_block_start();
        
        for(int b=0; b<N_B; b++)
        {
            if( w_block.count(pc->get_block_species(b)) == 0)
                throw_with_line_number("\"" + pc->get_block_species(b) + "\" species is not in w_block.");
        }

        for(const auto& item: w_block)
        {
            std::string species = item.first;
            double *w = item.second;
            for(int i=0; i<M; i++)
                exp_dw[species][i] = exp(-w[i]*ds);
        }
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                // initial condition
                if (q_init.count("1") == 0)
                {
                    for(int i=0; i<M; i++)
                        q_1[i] = exp_dw[pc->get_block_species(0)][i];
                }
                else
                {
                    for(int i=0; i<M; i++)
                        q_1[i] = exp_dw[pc->get_block_species(0)][i]*q_init["1"][i];
                }
                // apply the propagator successively
                for(int n=1; n<seg_start[1]; n++)
                    one_step(&q_1[(n-1)*M],&q_1[n*M], boltz_bond[0], exp_dw[pc->get_block_species(0)]);
                for(int b=1; b<N_B; b++)
                {
                    std::string species = pc->get_block_species(b);
                    // diffusion of segment between blocks
                    one_step(&q_1[(seg_start[b]-1)*M],&q_1[seg_start[b]*M],boltz_bond_middle[b-1],exp_dw[species]);
                    // diffusion of b-th block segments
                    for(int n=seg_start[b]+1; n<seg_start[b+1]; n++)
                        one_step(&q_1[(n-1)*M],&q_1[n*M],boltz_bond[b], exp_dw[species]);
                }
            }
            #pragma omp section
            {
                // initial condition
                if (q_init.count("2") == 0)
                {
                    for(int i=0; i<M; i++)
                        q_2[i+(N-1)*M] = exp_dw[pc->get_block_species(N_B-1)][i];
                }
                else
                {
                    for(int i=0; i<M; i++)
                        q_2[i+(N-1)*M] = exp_dw[pc->get_block_species(N_B-1)][i]*q_init["2"][i];
                }
                // apply the propagator successively

                for(int n=N-1; n>seg_start[N_B-1]; n--)
                    one_step(&q_2[n*M],&q_2[(n-1)*M],boltz_bond[N_B-1], exp_dw[pc->get_block_species(N_B-1)]);
                for(int b=N_B-1; b>0; b--)
                {
                    std::string species = pc->get_block_species(b-1);
                    // diffusion of segment between blocks
                    one_step(&q_2[seg_start[b]*M],&q_2[(seg_start[b]-1)*M],boltz_bond_middle[b-1],exp_dw[species]);
                    // diffusion of (b-1)th block segment
                    for(int n=seg_start[b]-1; n>0; n--)
                        one_step(&q_2[n*M],&q_2[(n-1)*M],boltz_bond[b-1], exp_dw[species]);
                }
            }
        }

        // calculates the single chain partition function
        if (q_init.count("2") == 0)
        {
            double q_2_init[M];
            for(int i=0; i<M; i++)
                q_2_init[i] = 1.0;
            single_partition = cb->inner_product(&q_1[(N-1)*M], q_2_init);
        }
        else
        {
            single_partition = cb->inner_product(&q_1[(N-1)*M], q_init["2"]);
        }

        // Compute segment concentration of each blocks
        for(int b=0; b<N_B; b++)
        {
            for(int i=0; i<M; i++)
                phi[b*M+i] = q_1[i+seg_start[b]*M]*q_2[i+seg_start[b]*M];
            for(int n=seg_start[b]+1; n<seg_start[b+1]; n++)
            {
                for(int i=0; i<M; i++)
                    phi[b*M+i] += q_1[i+n*M]*q_2[i+n*M];
            }
        }
        
        // normalize the concentration
        for(int b=0; b<N_B; b++)
        {
            for(int i=0; i<M; i++)
                phi[b*M+i] *= cb->get_volume()*pc->get_ds()/exp_dw[pc->get_block_species(b)][i]/single_partition;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoLinearDiscrete::one_step(double *q_in, double *q_out,
                                 double *boltz_bond, double *exp_dw)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = this->n_complex_grid;

        std::complex<double> k_q_in[M_COMPLEX];
        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= boltz_bond[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in,q_out);
        // normalization calculation and evaluate e^(-w*ds/2) in real space
        for(int i=0; i<M; i++)
            q_out[i] *= exp_dw[i];
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuPseudoLinearDiscrete::get_partition(double *q_out, int v, int u, int n)
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

        if (n < 1 || n > N)
            throw_with_line_number("n (" + std::to_string(n) + ") must be in range [1, " + std::to_string(N) + "]");

        if (v < u)
        {
            for(int i=0; i<M; i++)
                q_out[i] = q_1[(block_start[b]+n-1)*M+i];
        }
        else
        {
            for(int i=0; i<M; i++)
                q_out[i] = q_2[(N-block_start[b]-n)*M+i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}