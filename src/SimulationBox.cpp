/*-------------------------------------------------------------
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include "SimulationBox.h"

//----------------- Constructor -----------------------------
SimulationBox::SimulationBox(std::array<int,3> nx, std::array<double,3> lx)
{
    for(int i=0; i<3; i++)
    {
        this->nx[i] = nx[i];
        this->lx[i] = lx[i];
        this->dx[i] = lx[i]/nx[i];
    }
    // the number of total grids
    total_grids = nx[0]*nx[1]*nx[2];
    // weight factor for integral
    dv = new double[total_grids];
    for(int i=0; i<total_grids; i++)
        dv[i] = dx[0]*dx[1]*dx[2];
    // system polymer
    volume = lx[0]*lx[1]*lx[2];
}
//----------------- Destructor -----------------------------
SimulationBox::~SimulationBox()
{
    delete[] dv;
}
//-----------------------------------------------------------
// This method calculates integral g*h
double SimulationBox::dot(double *g, double *h)
{
    double sum{0.0};
    for(int i=0; i<total_grids; i++)
        sum += dv[i]*g[i]*h[i];
    return sum;
}
//-----------------------------------------------------------
double SimulationBox::multidot(int n_comp, double *g, double *h)
{
    double sum{0.0};
    for(int n=0; n<n_comp; n++)
    {
        for(int i=0; i<total_grids; i++)
            sum += dv[i]*g[i+n*total_grids]*h[i+n*total_grids];
    }
    return sum;
}
//-----------------------------------------------------------
// This method makes the input a zero-meaned matrix
void SimulationBox::zeromean(double *w)
{
    double sum{0.0};
    for(int i=0; i<total_grids; i++)
        sum += dv[i]*w[i];
    for(int i=0; i<total_grids; i++)
        w[i] -= sum/volume;
}
