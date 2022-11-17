#include <iostream>
#include <cmath>
#include <algorithm>

#include "PolymerChain.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
//PolymerChain::PolymerChain(std::vector<int> n_segment, std::vector<double> bond_length, double ds, std::string model_name)
PolymerChain::PolymerChain(std::vector<std::string> types, std::vector<double> block_lengths,
    std::map<std::string, double> dict_segment_lengths, double ds, std::string model_name)

{
    // check chain model
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

    // check block size
    if( types.size() != block_lengths.size())
        throw_with_line_number("The sizes of types (" +std::to_string(types.size()) + 
            ") and block_lengths (" +std::to_string(block_lengths.size()) + ") must be consistent");

    // check block lengths
    for(int i=0; i<block_lengths.size(); i++)
    {
        if( block_lengths[i] <= 0)
            throw_with_line_number("The block_length[" + std::to_string(i) + "] (" +std::to_string(block_lengths[i]) + ") must be a positive number");
    }

    // check block segments
    for(int i=0; i<block_lengths.size(); i++)
    {
        if( std::abs(std::lround(block_lengths[i]/ds)-block_lengths[i]/ds) > 1.e-6)
            throw_with_line_number("block_lengths[" + std::to_string(i) + "]/ds (" + std::to_string(block_lengths[i]) + "/" + std::to_string(ds) + ") is not an integer");
    }

    // save variable
    try
    {
        this->n_segment_total = 0;
        this->block_start = {0};
        for(int i=0; i<block_lengths.size(); i++)
        {
            this->segment_lengths.push_back(dict_segment_lengths[types[i]]*dict_segment_lengths[types[i]]); //bond length is stored as its as (a_A/a_Ref)^2.
            this->n_segments.push_back(std::lround(block_lengths[i]/ds));
            this->n_segment_total += std::lround(block_lengths[i]/ds);
            block_start.push_back(block_start.back() + n_segments[i]);
        }
        this->n_block = block_lengths.size();
        this->types = types;
        this->ds = ds;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
int PolymerChain::get_n_block()
{
    return n_block;
}
std::vector<int> PolymerChain::get_n_segment()
{
    return n_segments;
}
int PolymerChain::get_n_segment(int block)
{
    return n_segments[block];
}
int PolymerChain::get_n_segment_total()
{
    return n_segment_total;
}
double PolymerChain::get_ds()
{
    return ds;
}
std::vector<double> PolymerChain::get_bond_length()
{
    return segment_lengths;
}
double PolymerChain::get_bond_length(int block)
{
    return segment_lengths[block];
}

std::vector<std::string> PolymerChain::get_type()
{
    return types;
}
std::string PolymerChain::get_type(int block)
{
    return types[block];
}
std::string PolymerChain::get_model_name()
{
    return model_name;
}
std::vector<int> PolymerChain::get_block_start()
{
    return block_start;
}