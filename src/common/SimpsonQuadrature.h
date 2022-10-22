
#ifndef SIMPSON_QUADRATURE_H_
#define SIMPSON_QUADRATURE_H_

#include <cassert>

class SimpsonQuadrature
{
public:
    static void init_coeff(double *coeff, const int N)
    {
        assert(N >= 0 && "N must be a non-negative number");
        if ( N == 0)
            coeff[0] = 1.0;
        // Initialize the coefficients for the trapezoid rule.
        else if ( N == 1) 
        {
            coeff[0] = 1.0/2.0;
            coeff[1] = 1.0/2.0;
        }
        // Initialize the coefficients for the Simpson's 1/3 rule.
        else if ( N % 2 == 0) // when the number of segments is an even number
        {
            coeff[0] = 1.0/3.0;
            for(int n=1; n<=N-1; n++)
            {
                if ( n % 2 == 1)
                    coeff[n] = 4.0/3.0;
                else
                    coeff[n] = 2.0/3.0;
            }
            coeff[N] = 1.0/3.0;
        }
        else // when the number of segments is an odd number, use the Simpson's 3/8 rule for the last three points.
        {
            coeff[0] = 1.0/3.0;
            for(int n=1; n<=N-4; n++)
            {
                if ( n % 2 == 1)
                    coeff[n] = 4.0/3.0;
                else
                    coeff[n] = 2.0/3.0;
            }
            coeff[N]   = 3.0/8.0;
            coeff[N-1] = 9.0/8.0;
            coeff[N-2] = 9.0/8.0;
            coeff[N-3] = 3.0/8.0 + 1.0/3.0;
        }
    };
};
#endif