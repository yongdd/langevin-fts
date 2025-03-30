/*-------------------------------------------------------------
* This is an abstract Fast Fourier Transform (FFT) class
*------------------------------------------------------------*/

#ifndef FFT_H_
#define FFT_H_

#include <array>
#include <complex>
#include "Exception.h"

template <typename T>
class FFT
{
public:
    virtual ~FFT() {};
    virtual void forward (T *rdata, std::complex<double> *cdata) = 0;
    virtual void backward(std::complex<double> *cdata, T *rdata) = 0;
};

#endif
