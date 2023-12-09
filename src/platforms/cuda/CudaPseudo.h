/*----------------------------------------------------------
* This class defines a derived class for pseudo-spectral method
*-----------------------------------------------------------*/

#ifndef CUDA_PSEUDO_H_
#define CUDA_PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"
#include "Pseudo.h"

class CudaSolver : public Pseudo
{
private:

public:
    CudaSolver(ComputationBox *cb);
    ~CudaSolver() {};
    virtual void update_bond_function() = 0;
};
#endif
