// Original Author : ‪Tom Beardsley and Mark W Matsen‬ at University of Waterloo

//============================================================
// The first set of subroutines are used to generate random
// numbers with a uniform distribution in the range [0,1),
// using the MERSENNE TWISTER RANDOM NUMBER GENERATOR method.
// The final subroutine converts this to a normal distribution.
//------------------------------------------------------------

#ifndef RANDOM_GAUSSIAN_H_
#define RANDOM_GAUSSIAN_H_

#define BB 624
#define CC 397
#define MATRIX_A 0x9908b0dfUL /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))

class RandomGaussian
{
private:
    unsigned long state[BB]; /* the array for the state vector  */
    int left = 1;
    unsigned long *next;
    float y2;
    int use_last = 0;

    void next_state(void);
    unsigned long genrand_int32(void);
    double genrand_real2(void);

public:
    /* a default initial seed is used */
    RandomGaussian(unsigned long s=5489UL);
    double normal_dist(double mu, double sigma);
};
#endif
