/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

class Pseudo
{
public:
    virtual void find_phi(
        double *phia,  double *phib,
        double *q1_init, double *q2_init,
        double *wa, double *wb,
        double ds, double &QQ) {};

};
#endif
