#include <iostream>
#include <cctype>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <stack>
#include <set>

#include "Molecules.h"
#include "PropagatorCode.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
Molecules::Molecules(
    std::string model_name, double ds, std::map<std::string, double> bond_lengths)
{
    // Checking chain model
    std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                   [](unsigned char c)
    {
        return std::tolower(c);
    });

    if (model_name != "continuous" && model_name != "discrete")
    {
        throw_with_line_number(model_name + " is an invalid chain model. This must be 'Continuous' or 'Discrete'.");
    }

    // Save variables
    try
    {
        this->ds = ds;
        this->bond_lengths = bond_lengths;
        this->model_name = model_name;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void Molecules::add_polymer(
    double volume_fraction,
    std::vector<BlockInput> block_inputs,
    std::map<int, std::string> chain_end_to_q_init)
{
    // Add new polymer type
    polymer_types.push_back(Polymer(ds, bond_lengths, 
        volume_fraction, block_inputs, chain_end_to_q_init));
    Polymer& pc = polymer_types.back();
}
std::string Molecules::get_model_name() const
{
    return model_name;
}
double Molecules::get_ds() const
{
    return ds;
}
int Molecules::get_n_polymer_types() const
{
    return polymer_types.size();
}
Polymer& Molecules::get_polymer(const int p)
{
    return polymer_types[p];
}
const std::map<std::string, double>& Molecules::get_bond_lengths() const
{
    return bond_lengths;
}