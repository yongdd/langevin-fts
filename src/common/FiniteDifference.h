/*----------------------------------------------------------
* This class contains static methods for finite difference
*-----------------------------------------------------------*/

#ifndef FINITE_DIFFERENCE_H_
#define FINITE_DIFFERENCE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

class FiniteDifference
{
public:
    static void get_laplacian_matrix(
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx,
        double *xl, double *xd, double *xh,
        double *yl, double *yd, double *yh,
        double *zl, double *zd, double *zh,
        double bond_length_sq, double ds);
};
#endif