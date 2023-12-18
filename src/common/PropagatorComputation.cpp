#include <iostream>
#include <cmath>
#include "PropagatorComputation.h"

PropagatorComputation::PropagatorComputation(
    ComputationBox *cb,
    Molecules *molecules,
    PropagatorAnalyzer *propagator_analyzer)
{
    if (cb == nullptr)
        throw_with_line_number("ComputationBox *cb is a null pointer");
    if (molecules == nullptr)
        throw_with_line_number("Molecules *molecules is a null pointer");

    this->cb = cb;
    this->molecules = molecules;
    this->propagator_analyzer = propagator_analyzer;
}