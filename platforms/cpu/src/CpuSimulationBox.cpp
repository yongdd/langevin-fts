/*-------------------------------------------------------------
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include "CpuSimulationBox.h"

//----------------- Constructor -----------------------------
CpuSimulationBox::CpuSimulationBox(std::array<int,3> nx, std::array<double,3> lx)
{
    for(int i=0; i<3; i++)
    {
        this->nx[i] = nx[i];
        this->lx[i] = lx[i];
        this->dx[i] = lx[i]/nx[i];
    }
    // the number of total grids
    MM = nx[0]*nx[1]*nx[2];
    // weight factor for integral
    dv = new double[MM];
    for(int i=0; i<MM; i++)
        dv[i] = dx[0]*dx[1]*dx[2];
    // system polymer
    volume = lx[0]*lx[1]*lx[2];
}
//----------------- Destructor -----------------------------
CpuSimulationBox::~CpuSimulationBox()
{
    delete[] dv;
}
//-----------------------------------------------------------
// This method calculates integral g*h
double CpuSimulationBox::dot(double *g, double *h)
{
    double sum{0.0};
    for(int i=0; i<MM; i++)
        sum += dv[i]*g[i]*h[i];
    return sum;
}
//-----------------------------------------------------------
double CpuSimulationBox::multi_dot(int n_comp, double *g, double *h)
{
    double sum{0.0};
    for(int n=0; n<n_comp; n++)
    {
        for(int i=0; i<MM; i++)
            sum += dv[i]*g[i+n*MM]*h[i+n*MM];
    }
    return sum;
}
//-----------------------------------------------------------
// This method makes the input a zero-meaned matrix
void CpuSimulationBox::zero_mean(double *w)
{
    double sum{0.0};
    for(int i=0; i<MM; i++)
        sum += dv[i]*w[i];
    for(int i=0; i<MM; i++)
        w[i] -= sum/volume;
}
