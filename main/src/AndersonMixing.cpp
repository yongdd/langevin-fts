#include "AndersonMixing.h"
/*
void AndersonMixing::find_an(double **u, double *v, double *a, int n)
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
}*/
