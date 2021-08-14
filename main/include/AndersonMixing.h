/*-------------------------------------------------------------
* This is an abstract AndersonMixing class
*------------------------------------------------------------*/

#ifndef ANDERSON_MIXING_H_
#define ANDERSON_MIXING_H_

#include "SimulationBox.h"

class AndersonMixing
{
protected:
    void find_an(double **u, double *v, double *a, int n)
    {
        int i,j,k;
        double factor, tempsum;
        // elimination process
        for(i=0; i<n; i++)
        {
            for(j=i+1; j<n; j++)
            {
                factor = u[j][i]/u[i][i];
                v[j] = v[j] - v[i]*factor;
                for(k=i+1; k<n; k++)
                {
                    u[j][k] = u[j][k] - u[i][k]*factor;
                }
            }
        }
        // find the solution
        a[n-1] = v[n-1]/u[n-1][n-1];
        for(i=n-2; i>=0; i--)
        {
            tempsum = 0.0;
            for(j=i+1; j<n; j++)
            {
                tempsum = tempsum + u[i][j]*a[j];
            }
            a[i] = (v[i] - tempsum)/u[i][i];
        }
    };

public:
    
    virtual void reset_count() {};
    virtual void caculate_new_fields(
        double *w, double *w_out, double *w_diff,
        double old_error_level, double error_level) {};
};
#endif
