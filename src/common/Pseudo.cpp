#include <iostream>
#include <cmath>
#include "Pseudo.h"

//----------------- Constructor -----------------------------
template <typename T>
Pseudo<T>::Pseudo(
    std::map<std::string, double> bond_lengths,
    std::vector<BoundaryCondition> bc,
    std::vector<int> nx, std::vector<double> dx, double ds)
{
    try
    {
        this->bond_lengths = bond_lengths;
        this->bc = bc;
        this->nx = nx;
        this->dx = dx;
        this->ds = ds;

        update_total_complex_grid();
        const int M_COMPLEX = get_total_complex_grid();

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: bond_lengths)
        {
            std::string monomer_type = item.first;
            boltz_bond     [monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[monomer_type] = new double[M_COMPLEX]; 
        }

        // Allocate memory for stress calculation: compute_stress()
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];

        if constexpr (std::is_same<T, std::complex<double>>::value)
            k_idx = new int[M_COMPLEX];

        update_boltz_bond();
        update_weighted_fourier_basis();
        update_negative_frequency_mapping();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
//----------------- Destructor -----------------------------
template <typename T>
Pseudo<T>::~Pseudo()
{
    delete[] fourier_basis_x;
    delete[] fourier_basis_y;
    delete[] fourier_basis_z;
    if constexpr (std::is_same<T, std::complex<double>>::value)
        delete[] k_idx;

    for(const auto& item: boltz_bond)
        delete[] item.second;
    for(const auto& item: boltz_bond_half)
        delete[] item.second;
}
template <typename T>
int Pseudo<T>::get_total_complex_grid()
{
    return total_complex_grid;
}
template <typename T>
double* Pseudo<T>::get_boltz_bond(std::string monomer_type)
{
    return boltz_bond[monomer_type];
}
template <typename T>
double* Pseudo<T>::get_boltz_bond_half(std::string monomer_type)
{
    return boltz_bond_half[monomer_type];
}
template <typename T>
const double* Pseudo<T>::get_fourier_basis_x()
{
    return fourier_basis_x;
}
template <typename T>
const double* Pseudo<T>::get_fourier_basis_y()
{
    return fourier_basis_y;
}
template <typename T>
const double* Pseudo<T>::get_fourier_basis_z()
{
    return fourier_basis_z;
}
template <typename T>
const int* Pseudo<T>::get_negative_frequency_mapping()
{
    return k_idx;
}

//----------------- update_boltz_bond -------------------
template <typename T>
void Pseudo<T>::update_total_complex_grid()
{
    std::vector<int> nx = this->nx;
    int total_complex_grid;
    if constexpr (std::is_same<T, double>::value)
    {
        if (nx.size() == 3)
            total_complex_grid = nx[0]*nx[1]*(nx[2]/2+1);
        else if (nx.size() == 2)
            total_complex_grid = nx[0]*(nx[1]/2+1);
        else if (nx.size() == 1)
            total_complex_grid = nx[0]/2+1;
    }
    else
    {
        if (nx.size() == 3)
            total_complex_grid = nx[0]*nx[1]*nx[2];
        else if (nx.size() == 2)
            total_complex_grid = nx[0]*nx[1];
        else if (nx.size() == 1)
            total_complex_grid = nx[0];
    }
    this->total_complex_grid = total_complex_grid;
}

//----------------- update_boltz_bond -------------------
template <typename T>
void Pseudo<T>::update_boltz_bond()
{
    try
    {
        int itemp, jtemp, ktemp, idx;
        const double PI{3.14159265358979323846};
        double xfactor[3];

        const int DIM = nx.size();
        std::vector<int> tnx;
        std::vector<double> tdx;
        if (DIM == 3)
        {
            tnx = {nx[0], nx[1], nx[2]};
            tdx = {dx[0], dx[1], dx[2]};
        }
        else if (DIM == 2)
        {
            tnx = {1,   nx[0], nx[1]};
            tdx = {1.0, dx[0], dx[1]};
        }
        else if (DIM == 1)
        {
            tnx = {1,   1,   nx[0]};
            tdx = {1.0, 1.0, dx[0]};
        }

        for(size_t i=0; i<bc.size(); i++)
        {
            if (bc[i] != BoundaryCondition::PERIODIC)
                throw_with_line_number("Currently, pseudo-spectral method only supports periodic boundary conditions");
        }

        for(const auto& item: this->bond_lengths)
        {
            std::string monomer_type = item.first;
            double bond_length_sq    = item.second*item.second;
            double* _boltz_bond      = boltz_bond[monomer_type];
            double* _boltz_bond_half = boltz_bond_half[monomer_type];
            double mag_q2;

            // Calculate the exponential factor
            for(int d=0; d<3; d++)
                xfactor[d] = -bond_length_sq*std::pow(2*PI/(tnx[d]*tdx[d]),2)*ds/6.0;

            for(int i=0; i<tnx[0]; i++)
            {
                if( i > tnx[0]/2)
                    itemp = tnx[0]-i;
                else
                    itemp = i;
                for(int j=0; j<tnx[1]; j++)
                {
                    if( j > tnx[1]/2)
                        jtemp = tnx[1]-j;
                    else
                        jtemp = j;

                    if constexpr (std::is_same<T, double>::value)
                    {
                        for(int k=0; k<tnx[2]/2+1; k++)
                        {
                            ktemp = k;
                            idx = i* tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;

                            mag_q2 = itemp*itemp*xfactor[0]+jtemp*jtemp*xfactor[1]+ktemp*ktemp*xfactor[2];
                            _boltz_bond     [idx] = exp(mag_q2);
                            _boltz_bond_half[idx] = exp(mag_q2/2.0);
                        }
                    }
                    else
                    {
                        for(int k=0; k<tnx[2]; k++)
                        {
                            if( k > tnx[2]/2)
                                ktemp = tnx[2]-k;
                            else
                                ktemp = k;
                            idx = i* tnx[1]*tnx[2] + j*tnx[2] + k;

                            mag_q2 = itemp*itemp*xfactor[0]+jtemp*jtemp*xfactor[1]+ktemp*ktemp*xfactor[2];
                            _boltz_bond     [idx] = exp(mag_q2);
                            _boltz_bond_half[idx] = exp(mag_q2/2.0);
                        }
                    }
                }
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void Pseudo<T>::update_weighted_fourier_basis()
{
    int itemp, jtemp, ktemp, idx;
    const double PI{3.14159265358979323846};
    double xfactor[3];
    
    const int DIM = nx.size();
    std::vector<int> tnx;
    std::vector<double> tdx;
    if (DIM == 3)
    {
        tnx = {nx[0], nx[1], nx[2]};
        tdx = {dx[0], dx[1], dx[2]};
    }
    else if (DIM == 2)
    {
        tnx = {1,   nx[0], nx[1]};
        tdx = {1.0, dx[0], dx[1]};
    }
    else if (DIM == 1)
    {
        tnx = {1,   1,   nx[0]};
        tdx = {1.0, 1.0, dx[0]};
    }

    for(size_t i=0; i<bc.size(); i++)
    {
        if (bc[i] != BoundaryCondition::PERIODIC)
            throw_with_line_number("Currently, pseudo-spectral method only supports periodic boundary conditions");
    }

    // Calculate the exponential factor
    for(int d=0; d<3; d++)
        xfactor[d] = std::pow(2*PI/(tnx[d]*tdx[d]),2);

    for(int i=0; i<tnx[0]; i++)
    {
        if( i > tnx[0]/2)
            itemp = tnx[0]-i;
        else
            itemp = i;
        for(int j=0; j<tnx[1]; j++)
        {
            if( j > tnx[1]/2)
                jtemp = tnx[1]-j;
            else
                jtemp = j;

            if constexpr (std::is_same<T, double>::value)
            {
                for(int k=0; k<tnx[2]/2+1; k++)
                {
                    ktemp = k;
                    idx = i* tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;
                    fourier_basis_x[idx] = itemp*itemp*xfactor[0];
                    fourier_basis_y[idx] = jtemp*jtemp*xfactor[1];
                    fourier_basis_z[idx] = ktemp*ktemp*xfactor[2];
                    if (k != 0 && 2*k != tnx[2])
                    {
                        fourier_basis_x[idx] *= 2;
                        fourier_basis_y[idx] *= 2;
                        fourier_basis_z[idx] *= 2;
                    }
                }
            }
            else
            {
                for(int k=0; k<tnx[2]; k++)
                {
                    if( k > tnx[2]/2)
                        ktemp = tnx[2]-k;
                    else
                        ktemp = k;
                    idx = i* tnx[1]*tnx[2] + j*tnx[2] + k;
                    fourier_basis_x[idx] = itemp*itemp*xfactor[0];
                    fourier_basis_y[idx] = jtemp*jtemp*xfactor[1];
                    fourier_basis_z[idx] = ktemp*ktemp*xfactor[2];
                }
            }
        }
    }
}

template <typename T>
void Pseudo<T>::update_negative_frequency_mapping(){}

template <>
void Pseudo<std::complex<double>>::update_negative_frequency_mapping()
{
    const int DIM = nx.size();
    int itemp, jtemp, ktemp, idx, idx_minus;
    std::vector<int> tnx;

    if (DIM == 3)
    {
        tnx = {nx[0], nx[1], nx[2]};
    }
    else if (DIM == 2)
    {
        tnx = {1,   nx[0], nx[1]};
    }
    else if (DIM == 1)
    {
        tnx = {1,   1,   nx[0]};
    }

    for(int i=0; i<tnx[0]; i++)
    {
        if(i == 0)
            itemp = 0;
        else
            itemp = tnx[0]-i;
        for(int j=0; j<tnx[1]; j++)
        {
            if(j == 0)
                jtemp = 0;
            else
                jtemp = tnx[1]-j;
            for(int k=0; k<tnx[2]; k++)
            {
                if(k == 0)
                    ktemp = 0;
                else
                    ktemp = tnx[2]-k;

                idx       = i    *tnx[1]*tnx[2] + j    *tnx[2] + k;
                idx_minus = itemp*tnx[1]*tnx[2] + jtemp*tnx[2] + ktemp;

                k_idx[idx] = idx_minus;

            //     0   0   0
            //     1   1   6
            //     2   2   5
            //     3   3   4
            //    -3   4   3
            //    -2   5   2
            //    -1   6   1

            //     0   0   0
            //     1   1   7
            //     2   2   6
            //     3   3   5
            //    -4   4   4
            //    -3   5   3
            //    -2   6   2
            //    -1   7   1

            }
        }
    }
    // const int M_COMPLEX = get_total_complex_grid();
    // for(int i=0;i<M_COMPLEX;i++)
    //     std::cout << i << ", " << k_idx[i] << ", " << std::endl;
}

// Explicit template instantiation
template class Pseudo<double>;
template class Pseudo<std::complex<double>>;
