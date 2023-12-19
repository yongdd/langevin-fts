#include <iostream>
#include <cmath>
#include "Pseudo.h"

int Pseudo::get_n_complex_grid(std::vector<int> nx)
{
    int n_complex_grid;
    if (nx.size() == 3)
        n_complex_grid = nx[0]*nx[1]*(nx[2]/2+1);
    else if (nx.size() == 2)
        n_complex_grid = nx[0]*(nx[1]/2+1);
    else if (nx.size() == 1)
        n_complex_grid = nx[0]/2+1;
    return n_complex_grid;
}
//----------------- get_boltz_bond -------------------
void Pseudo::get_boltz_bond(
    std::vector<BoundaryCondition> bc,
    double *boltz_bond, double bond_length_variance,
    std::vector<int> nx, std::vector<double> dx, double ds)
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

        // Calculate the exponential factor
        for(int d=0; d<3; d++)
            xfactor[d] = -bond_length_variance*std::pow(2*PI/(tnx[d]*tdx[d]),2)*ds/6.0;

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
                for(int k=0; k<tnx[2]/2+1; k++)
                {
                    ktemp = k;
                    idx = i* tnx[1]*(tnx[2]/2+1) + j*(tnx[2]/2+1) + k;
                    boltz_bond[idx] = exp(itemp*itemp*xfactor[0]+jtemp*jtemp*xfactor[1]+ktemp*ktemp*xfactor[2]);
                }
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void Pseudo::get_weighted_fourier_basis(
    std::vector<BoundaryCondition> bc,
    double *fourier_basis_x,
    double *fourier_basis_y,
    double *fourier_basis_z,
    std::vector<int> nx,
    std::vector<double> dx)
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
    }
}
