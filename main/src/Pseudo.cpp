#include "cmath"
#include "Pseudo.h"

Pseudo::Pseudo(
    SimulationBox *sb,
    PolymerChain *pc)
{
    this->sb = sb;
    this->MM         = sb->MM;
    this->MM_COMPLEX = sb->nx[0]*sb->nx[1]*(sb->nx[2]/2+1);
    this->NN = pc->NN;
    this->NNf= pc->NNf;
    this->ds = pc->ds;

    this->expf = new double[MM_COMPLEX];
    this->expf_half = new double[MM_COMPLEX];
    init_gaussian_factor(sb->nx, sb->dx, pc->ds);
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
