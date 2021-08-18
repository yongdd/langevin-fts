/*-------------------------------------------------------------
* This is an abstract Fast Fourier Transform (FFT) class
*------------------------------------------------------------*/

#ifndef FFT_H_
#define FFT_H_

#include <array>
#include <complex>

class FFT
{
public:
    virtual void forward (double *rdata, std::complex<double> *cdata){};
    virtual void backward(std::complex<double> *cdata, double *rdata){};
};
#endif
