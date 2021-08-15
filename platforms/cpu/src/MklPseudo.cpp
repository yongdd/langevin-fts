#include <cmath>
#include "MklPseudo.h"
#include "MklFFT.h"

MklPseudo::MklPseudo(
    SimulationBox *sb, 
    PolymerChain *pc)
    : Pseudo(sb, pc)
{
    this->fft = new MklFFT(sb->nx);
}
MklPseudo::~MklPseudo()
{
    delete fft;
}
void MklPseudo::find_phi(double *phia,  double *phib,
                         double *q1_init, double *q2_init,
                         double *wa, double *wb, double &QQ)
{
    Pseudo::find_phi(phia, phib, q1_init, q2_init, wa, wb, QQ);
}



