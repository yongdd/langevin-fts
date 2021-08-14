#include <cmath>
#include "MklPseudo.h"
#include "MklFFT.h"

MklPseudo::MklPseudo(
    SimulationBox *sb, double ds,
    int NN, int NNf)
    : Pseudo(sb, ds, NN, NNf)
{
    this->fft = new MklFFT(sb->nx);
}
MklPseudo::~MklPseudo()
{
    delete fft;
}
void MklPseudo::find_phi(double *phia,  double *phib,
                         double *q1_init, double *q2_init,
                         double *wa, double *wb, double ds, double &QQ)
{
    Pseudo::find_phi(phia, phib, q1_init, q2_init, wa, wb, ds, QQ);
}



