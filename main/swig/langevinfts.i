%module langevinfts

%{
#define SWIG_FILE_WITH_INIT
#include "ParamParser.h"
#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "RandomGaussian.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
%}

%include "std_string.i"
%include "std_array.i"
%include "std_vector.i"

%template(IntArray3) std::array<int,3>;
%template(DoubleArray3) std::array<double,3>;
%template(IntVector) std::vector<int>;
%template(DoubleVector) std::vector<double>;
%template(StringVector) std::vector<std::string>;

%include "numpy.i"
%init %{
import_array();
%}
%apply (double* INPLACE_ARRAY1, int DIM1){
(double *g, int len_g),
(double *h, int len_h),
(double *phia, int len_pa),
(double *phib, int len_pb),
(double *wa, int len_wa),
(double *wb, int len_wb),
(double *q1_init, int len_q1),
(double *q2_init, int len_q2),
(double *w_in, int len_w_in),
(double *w_out, int len_wout),
(double *w_diff, int len_wdiff)};

%apply double* OUTPUT {double &QQ};

%include "ParamParser.h"
%include "PolymerChain.h"
%include "SimulationBox.h"
%include "Pseudo.h"
%include "AndersonMixing.h"
%include "RandomGaussian.h"
%include "AbstractFactory.h"
%include "PlatformSelector.h"
