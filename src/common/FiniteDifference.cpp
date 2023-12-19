#include <iostream>
#include <cmath>
#include "FiniteDifference.h"

//----------------- get_laplacian_matrix -------------------
void FiniteDifference::get_laplacian_matrix(
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx,
        double *xl, double *xd, double *xh,
        double *yl, double *yd, double *yh,
        double *zl, double *zd, double *zh,
        double ds)
{
    try
    {
        int itemp, jtemp, ktemp, idx;
        double xfactor[3] = {0.0};

        const int DIM = nx.size();

        for(size_t i=0; i<bc.size(); i++)
        {
            if (bc[i] == BoundaryCondition::PERIODIC)
                throw_with_line_number("Currently, we do not support periodic boundary conditions in real-space method");
        }

        // Calculate the exponential factor
        for(int d=0; d<DIM; d++)
            xfactor[d] = ds/(std::pow(dx[d],2)*6.0);
        
        if(DIM >= 1)
        {
            for(int i=0; i<nx[0]; i++)
            {
                xl[i] = -0.5*xfactor[0];
                xd[i] =  1.0+xfactor[0];
                xh[i] = -0.5*xfactor[0];
            }
        }
        if(DIM >= 2)
        {
            for(int i=0; i<nx[1]; i++)
            {
                yl[i] = -0.5*xfactor[1];
                yd[i] =  1.0+xfactor[1];
                yh[i] = -0.5*xfactor[1];
            }
        }
        if(DIM >= 3)
        {
            for(int i=0; i<nx[2]; i++)
            {
                zl[i] = -0.5*xfactor[2];
                zd[i] =  1.0+xfactor[2];
                zh[i] = -0.5*xfactor[2];
            }
        }

        for(int d=0; d<DIM; d++)
        {
            const int _nx_max = nx[d]-1;
            BoundaryCondition bcl = bc[2*d];
            BoundaryCondition bch = bc[2*d+1];
            double *_xl;
            double *_xd;
            double *_xh;

            if(d == 0)
            {
                _xl = xl;
                _xd = xd;
                _xh = xh;
            }
            else if(d == 1)
            {
                _xl = yl;
                _xd = yd;
                _xh = yh;
            }
            else if(d == 2)
            {
                _xl = zl;
                _xd = zd;
                _xh = zh;
            }

            if(bcl == BoundaryCondition::REFLECTING)
            {
                _xd[0] += _xl[0];
                _xl[0] = 0.0;
            }
            else if(bcl == BoundaryCondition::ABSORBING)
            {
                _xl[0] = 0.0;
            }

            if(bch == BoundaryCondition::REFLECTING)
            {
                _xd[_nx_max] += _xh[_nx_max];
                _xh[_nx_max] = 0.0;
            }
            else if(bch == BoundaryCondition::ABSORBING)
            {
                _xh[_nx_max] = 0.0;
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}