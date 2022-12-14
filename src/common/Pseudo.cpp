#include <iostream>
#include <cmath>
#include "Pseudo.h"

Pseudo::Pseudo(
    ComputationBox *cb,
    Mixture *mx)
{
    if (cb == nullptr)
        throw_with_line_number("ComputationBox *cb is null pointer");
    if (mx == nullptr)
        throw_with_line_number("Mixture *mx is null pointer");

    this->cb = cb;
    this->mx = mx;
    this->n_complex_grid = cb->get_nx(0)*cb->get_nx(1)*(cb->get_nx(2)/2+1);
}
//----------------- get_boltz_bond -------------------
void Pseudo::get_boltz_bond(double *boltz_bond, double bond_length_variance,
                            std::array<int,3> nx, std::array<double,3> dx, double ds)
{
    int itemp, jtemp, ktemp, idx;
    double xfactor[3];
    const double PI{3.14159265358979323846};

    // calculate the exponential factor
    for(int d=0; d<3; d++)
        xfactor[d] = -std::pow(2*PI/(nx[d]*dx[d]),2)*ds/6.0;

    for(int i=0; i<nx[0]; i++)
    {
        if( i > nx[0]/2)
            itemp = nx[0]-i;
        else
            itemp = i;
        for(int j=0; j<nx[1]; j++)
        {
            if( j > nx[1]/2)
                jtemp = nx[1]-j;
            else
                jtemp = j;
            for(int k=0; k<nx[2]/2+1; k++)
            {
                ktemp = k;
                idx = i* nx[1]*(nx[2]/2+1) + j*(nx[2]/2+1) + k;
                boltz_bond[idx] = exp(bond_length_variance*
                    (pow(itemp,2)*xfactor[0]+pow(jtemp,2)*xfactor[1]+pow(ktemp,2)*xfactor[2]));
            }
        }
    }
}

void Pseudo::get_weighted_fourier_basis(
    double *fourier_basis_x,
    double *fourier_basis_y,
    double *fourier_basis_z,
    std::array<int,3> nx,
    std::array<double,3> dx)
{
    int itemp, jtemp, ktemp, idx;
    double xfactor[3];
    const double PI{3.14159265358979323846};

    // calculate the exponential factor
    for(int d=0; d<3; d++)
        xfactor[d] = std::pow(2*PI/(nx[d]*dx[d]),2);

    for(int i=0; i<nx[0]; i++)
    {
        if( i > nx[0]/2)
            itemp = nx[0]-i;
        else
            itemp = i;
        for(int j=0; j<nx[1]; j++)
        {
            if( j > nx[1]/2)
                jtemp = nx[1]-j;
            else
                jtemp = j;
            for(int k=0; k<nx[2]/2+1; k++)
            {
                ktemp = k;
                idx = i* nx[1]*(nx[2]/2+1) + j*(nx[2]/2+1) + k;
                fourier_basis_x[idx] = pow(itemp,2)*xfactor[0];
                fourier_basis_y[idx] = pow(jtemp,2)*xfactor[1];
                fourier_basis_z[idx] = pow(ktemp,2)*xfactor[2];
                if (k != 0 && 2*k != nx[2])
                {
                    fourier_basis_x[idx] *= 2;
                    fourier_basis_y[idx] *= 2;
                    fourier_basis_z[idx] *= 2;
                }
            }
        }
    }
}
