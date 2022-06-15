#include <iostream>
#include <cmath>
#include <algorithm>
#include "PolymerChain.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
PolymerChain::PolymerChain(double f, int n_segment,
    double chi_n, std::string model_name, double epsilon)
{
    if( f <= 0 || f >= 1)
        throw_with_line_number("A fraction f (" + std::to_string(f) + ") must be in range (0, 1)");
    if( n_segment <= 0)
        throw_with_line_number("The number of segments (" +std::to_string(n_segment) + ") must be a postive number");
    if( chi_n < 0)
        throw_with_line_number("The Flory-Hunggins parameter (" +std::to_string(chi_n) + ") must be a non-negative number");
    if( epsilon <= 0)
        throw_with_line_number("Conformational asymmetry (" +std::to_string(epsilon) + ") must be a postive number");

    this->f = f;
    this->n_segment = n_segment;
    this->chi_n = chi_n;
    this->epsilon = epsilon;

    // the number of A segments
    this->n_segment_a = std::lround(n_segment*f);
    if( std::abs(this->n_segment_a-n_segment*f) > 1.e-6)
        throw_with_line_number("N*f (" + std::to_string(n_segment*f) + ") is not an integer");
    // segment step size
    this->ds = 1.0/n_segment;

    // chain model
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char c)
    {
        return std::tolower(c);
    });
    if (model_name != "continuous" && model_name != "discrete")
    {
        throw_with_line_number(model_name + " is an invalid chain model. This must be 'Continuous' or 'Discrete'");
    }
    this->model_name = model_name;
}
int PolymerChain::get_n_segment()
{
    return n_segment;
}
int PolymerChain::get_n_segment_a()
{
    return n_segment_a;
}
int PolymerChain::get_n_segment_b()
{
    return n_segment - n_segment_a;
}
double PolymerChain::get_f()
{
    return f;
}
double PolymerChain::get_ds()
{
    return ds;
}
double PolymerChain::get_chi_n()
{
    return chi_n;
}
double PolymerChain::get_epsilon()
{
    return epsilon;
}
std::string PolymerChain::get_model_name()
{
    return model_name;
}
void PolymerChain::set_chi_n(double chi_n)
{
    if( chi_n < 0)
        throw_with_line_number("The Flory-Hunggins parameter (" +std::to_string(chi_n) + ") must be a non-negative number");
    this->chi_n = chi_n;
}
