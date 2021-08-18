// Original Author : ‪Tom Beardsley and Mark W Matsen‬ at University of Waterloo

#include <cmath>
#include "RandomGaussian.h"

//============================================================
// initializes state[BB] with a seed
//------------------------------------------------------------
RandomGaussian::RandomGaussian(unsigned long s)
{
    int j;
    state[0] = s & 0xffffffffUL;
    for (j = 1; j<BB; j++)
    {
        state[j] = (1812433253UL * (state[j - 1] ^ (state[j - 1] >> 30)) + j);
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array state[].                     */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        state[j] &= 0xffffffffUL;  /* for >32 bit machines */
    }
    left = 1;
}
//============================================================
void RandomGaussian::next_state(void)
{
    unsigned long *p = state;
    int j;

    left = BB;
    next = state;

    for (j = BB - CC + 1; --j; p++)
        *p = p[CC] ^ TWIST(p[0], p[1]);

    for (j = CC; --j; p++)
        *p = p[CC - BB] ^ TWIST(p[0], p[1]);

    *p = p[CC - BB] ^ TWIST(p[0], state[0]);
}
//============================================================
// generates a random number on [0,0xffffffff]-interval
//------------------------------------------------------------
unsigned long RandomGaussian::genrand_int32(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
//============================================================
// generates a random number on [0,1)-real-interval
//------------------------------------------------------------
double RandomGaussian::genrand_real2(void)
{
    unsigned long y;

    if (--left == 0) next_state();
    y = *next++;

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (double)y * (1.0 / 4294967296.0);
    /* divided by 2^32 */
}
//============================================================
// Generates a normal-distributed random number from a uniform
// distribution in the range [-1,1)
// The algorithm is from https://www.taygeta.com/random/boxmuller
//------------------------------------------------------------
double RandomGaussian::normal_dist(double mu, double sigma)
{
    double x1, x2, w, y1;
    if (use_last)  // use value from previous call
    {
        y1 = y2;
        use_last = 0;
    }
    else
    {
        do
        {
            x1 = 2.0*genrand_real2()-1.0;  // uniform distribution on [-1,1)
            x2 = 2.0*genrand_real2()-1.0;  // uniform distribution on [-1,1)
            w = x1 * x1 + x2 * x2;
        }
        while (w >= 1.0);

        w = sqrt(-2.0*log(w)/w);
        y1 = x1 * w;            // Normal distribution with mu=0, sigma=1
        y2 = x2 * w;            // Normal distribution with mu=0, sigma=1
        use_last = 1;
    }
    return mu+y1*sigma;
}
