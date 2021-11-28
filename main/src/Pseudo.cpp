
#include <iostream>
#include "cmath"
#include "Pseudo.h"

Pseudo::Pseudo(
    SimulationBox *sb,
    PolymerChain *pc)
{
    this->sb = sb;
    this->pc = pc;

    if(sb->get_dim()==3)
        this->n_complex_grid = sb->get_nx(0)*sb->get_nx(1)*(sb->get_nx(2)/2+1);
    else if(sb->get_dim()==2)
        this->n_complex_grid = sb->get_nx(0)*(sb->get_nx(1)/2+1);
    else if(sb->get_dim()==1)
        this->n_complex_grid = sb->get_nx(0)/2+1;
    else
        std::cerr << "Pseudo: Invalid dimension " << sb->get_dim() << std::endl;
    
    this->expf = new double[n_complex_grid];
    this->expf_half = new double[n_complex_grid];
    init_gaussian_factor(sb->get_nx(), sb->get_dx(), pc->get_ds());
}
Pseudo::~Pseudo()
{
    delete[] expf;
    delete[] expf_half;
}
//----------------- init_gaussian_factor -------------------
void Pseudo::init_gaussian_factor(
    std::array<int,3> nx, std::array<double,3> dx, double ds)
{
    int itemp, jtemp, ktemp, idx;
    double xfactor[3];
    const double PI{3.14159265358979323846};

    // calculate the exponential factor
    for(int d=0; d<3; d++)
        xfactor[d] = -std::pow(2*PI/(nx[d]*dx[d]),2)*ds/6.0;

    if(sb->get_dim()==3)
    {
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
                    expf[idx] = exp(pow(itemp,2)*xfactor[0]+pow(jtemp,2)*xfactor[1]+pow(ktemp,2)*xfactor[2]);
                    expf_half[idx] = exp((pow(itemp,2)*xfactor[0]+pow(jtemp,2)*xfactor[1]+pow(ktemp,2)*xfactor[2])/2);
                }
            }
        }
    }
    else if (sb->get_dim()==2)
    {
        for(int i=0; i<nx[0]; i++)
        {
            if( i > nx[0]/2)
                itemp = nx[0]-i;
            else
                itemp = i;
            for(int j=0; j<nx[1]/2+1; j++)
            {
                jtemp = j;
                idx = i* (nx[1]/2+1) + j;
                expf[idx] = exp(pow(itemp,2)*xfactor[0]+pow(jtemp,2)*xfactor[1]);
                expf_half[idx] = exp((pow(itemp,2)*xfactor[0]+pow(jtemp,2)*xfactor[1])/2);

            }
        }
    }
    else if (sb->get_dim()==1)
    {
        for(int i=0; i<nx[0]/2+1; i++)
        {
            expf[i] = exp(pow(i,2)*xfactor[0]);
            expf_half[i] = exp((pow(i,2)*xfactor[0])/2);

        }
    }
}
