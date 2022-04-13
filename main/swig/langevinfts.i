%module langevinfts

%{
#define SWIG_FILE_WITH_INIT
#include "ParamParser.h"
#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
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
%apply (double *IN_ARRAY1, int DIM1){
(double *q1_init, int len_q1),
(double *q2_init, int len_q2),
(double *w_a, int len_w_a),
(double *w_b, int len_w_b)};
%apply (double *INPLACE_ARRAY1, int DIM1){
(double *g, int len_g),
(double *h, int len_h),
(double *w_in, int len_w_in),
(double *w_out, int len_w_out),
(double *w_diff, int len_w_diff)};
%apply (double **ARGOUTVIEWM_ARRAY1, int *DIM1){
(double **phi_a, int *len_p_a),
(double **phi_b, int *len_p_b),
(double **q1_out, int *len_q1),
(double **q2_out, int *len_q2)};

%apply double* OUTPUT {double &single_partition};

%include "ParamParser.h"
%include "PolymerChain.h"
%include "SimulationBox.h"
%include "Pseudo.h"
%include "AndersonMixing.h"
%include "AbstractFactory.h"
%include "PlatformSelector.h"
