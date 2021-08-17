%module langevinfts

%{
#define SWIG_FILE_WITH_INIT
#include "ParamParser.h"
#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "KernelFactory.h"
%}

%include "std_string.i"
%include "std_array.i"
%include "std_vector.i"

%template(IntArray3) std::array<int,3>;
%template(DoubleArray3) std::array<double,3>;
%template(StringVector) std::vector<std::string>;

%include "numpy.i"
%init %{
import_array();
%}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* g, int len_c)}

%include "ParamParser.h"
%include "PolymerChain.h"
%include "SimulationBox.h"
%include "Pseudo.h"
%include "AndersonMixing.h"
%include "KernelFactory.h"

