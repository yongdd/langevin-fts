#include "FftwPseudo.h"
#include "FftwFFT.h"

FftwPseudo::FftwPseudo(
    SimulationBox *sb, double ds,
    int NN, int NNf)
    : Pseudo(sb, ds, NN, NNf)
{
    this->fft = new FftwFFT(sb->nx);
}
FftwPseudo::~FftwPseudo()
{
    delete fft;
}
void FftwPseudo::find_phi(double *phia,  double *phib,
                         double *q1_init, double *q2_init,
                         double *wa, double *wb, double ds, double &QQ)
{
    Pseudo::find_phi(phia, phib, q1_init, q2_init, wa, wb, ds, QQ);
}



