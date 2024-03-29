
#ifndef SIMPSON_RULE_H_
#define SIMPSON_RULE_H_

#include <cassert>

class SimpsonRule
{
public:
    static std::vector<double> get_coeff(const int N)
    {
        assert(N >= 0 && "N must be a non-negative number");
        std::vector<double> coeff;
        if ( N == 0)
            coeff.push_back(1.0);
        // Initialize the coefficients for the trapezoid rule.
        else if ( N == 1) 
        {
            coeff.push_back(1.0/2.0);
            coeff.push_back(1.0/2.0);
        }
        // Initialize the coefficients for the Simpson's 3/8 rule.
        else if ( N == 3) 
        {
            coeff.push_back(3.0/8.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(3.0/8.0);
        }
        // Initialize the coefficients for the Simpson's 1/3 rule.
        else if ( N % 2 == 0) // when the number of segments is an even number
        {
            coeff.push_back(1.0/3.0);
            for(int n=1; n<=N-1; n++)
            {
                if ( n % 2 == 1)
                    coeff.push_back(4.0/3.0);
                else
                    coeff.push_back(2.0/3.0);
            }
            coeff.push_back(1.0/3.0);
        }
        else // when the number of segments is an odd number, use the Simpson's 3/8 rule for the last three points.
        {
            coeff.push_back(1.0/3.0);
            for(int n=1; n<=N-4; n++)
            {
                if ( n % 2 == 1)
                    coeff.push_back(4.0/3.0);
                else
                    coeff.push_back(2.0/3.0);
            }
            coeff.push_back(3.0/8.0 + 1.0/3.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(3.0/8.0);
        }
        return coeff;
    };
};
#endif