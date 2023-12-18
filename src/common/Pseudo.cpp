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
void Pseudo::get_boltz_bond(double *boltz_bond, double bond_length_variance,
                            std::vector<int> nx, std::vector<double> dx, double ds)
{
    int itemp, jtemp, ktemp, idx;
    const double PI{3.14159265358979323846};
    double xfactor[3];

    const int DIM = nx.size();
    std::vector<int> tnx = {1,  1,  1};
    std::vector<double> tdx = {1.0,1.0,1.0};
    for(int d=0; d<DIM; d++)
    {
        tnx[3-DIM+d] = nx[d];
        tdx[3-DIM+d] = dx[d];
    }

    // Calculate the exponential factor
    for(int d=0; d<3; d++)
        xfactor[d] = -std::pow(2*PI/(tnx[d]*tdx[d]),2)*ds/6.0;

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
                boltz_bond[idx] = exp(bond_length_variance*
                    (itemp*itemp*xfactor[0]+jtemp*jtemp*xfactor[1]+ktemp*ktemp*xfactor[2]));
            }
        }
    }
}

void Pseudo::get_weighted_fourier_basis(
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
    std::vector<int> tnx = {1,  1,  1};
    std::vector<double> tdx = {1.0,1.0,1.0};
    for(int d=0; d<DIM; d++)
    {
        tnx[3-DIM+d] = nx[d];
        tdx[3-DIM+d] = dx[d];
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
