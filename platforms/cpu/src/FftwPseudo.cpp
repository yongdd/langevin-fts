#include "FftwPseudo.h"
#include "FftwFFT.h"

FftwPseudo::FftwPseudo(
    SimulationBox *sb,  
    PolymerChain *pc)
    : Pseudo(sb, pc)
{
    this->fft = new FftwFFT(sb->nx);
}
FftwPseudo::~FftwPseudo()
{
    delete fft;
}
void FftwPseudo::find_phi(double *phia,  double *phib,
                         double *q1_init, double *q2_init,
                         double *wa, double *wb, double &QQ)
{
    Pseudo::find_phi(phia, phib, q1_init, q2_init, wa, wb, QQ);
}



