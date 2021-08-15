/*-------------------------------------------------------------
* This is a derived MklPseudo class
*------------------------------------------------------------*/

#ifndef MKL_PSEUDO_H_
#define MKL_PSEUDO_H_

#include "Pseudo.h"

class MklPseudo : public Pseudo
{

public:

    MklPseudo(SimulationBox *sb, PolymerChain *pc);
    ~MklPseudo();
    
    void find_phi(double *phia,  double *phib,
                  double *q1_init, double *q2_init,
                  double *wa, double *wb, double &QQ) override;
};
#endif
