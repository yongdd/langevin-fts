#include <iostream>
#include <cmath>
#include <algorithm>
#include "PolymerChain.h"
#include "Exception.h"

//----------------- Constructor ----------------------------
PolymerChain::PolymerChain(std::vector<int> n_segment, std::vector<double> bond_length, std::string model_name)
{
   
    if( n_segment.size() != bond_length.size())
        throw_with_line_number("The number of blocks for n_segment (" +std::to_string(n_segment.size()) + ") and bond_length (" +std::to_string(bond_length.size()) + ")must be consistent");
    
    this->n_block = n_segment.size();
    for(int i=0; i<n_block; i++)
    {
        if( n_segment[i] <= 0)
            throw_with_line_number("The number of segments (" +std::to_string(n_segment[i]) + ") must be a postive number");
        if( bond_length[i] <= 0)
            throw_with_line_number("The bond length (" +std::to_string(n_segment[i]) + ") must be a postive number");
    }

    this->n_segment = n_segment;
    
    this->n_segment_total = 0;
    for(int i=0; i<n_block; i++)
    {
        this->n_segment_total += this->n_segment[i];
    }
    //bond length is stored as its square*N_total
    this->bond_length = bond_length;

    this->relative_length = 0;
    for(int i=0; i<n_block; i++)
    {
        this->relative_length += bond_length[i]*n_segment[i];
    }
    this->relative_length /= this->n_segment_total;

    // segment step size
    this->ds = this->relative_length/this->n_segment_total;
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

    this->block_start = {0};
    for(int i=0; i<n_block; i++)
    {
        block_start.push_back(block_start.back() + n_segment[i]);
    }  

}
int PolymerChain::get_n_block()
{
    return n_block;
}
std::vector<int> PolymerChain::get_n_segment()
{
    return n_segment;
}
int PolymerChain::get_n_segment(int block)
{
    return n_segment[block];
}
int PolymerChain::get_n_segment_total()
{
    return n_segment_total;
}
double PolymerChain::get_ds()
{
    return ds;
}
double PolymerChain::get_relative_length()
{
    return relative_length;
}
std::vector<double> PolymerChain::get_bond_length()
{
    return bond_length;
}
double PolymerChain::get_bond_length(int block)
{
    return bond_length[block];
}
std::string PolymerChain::get_model_name()
{
    return model_name;
}
std::vector<int> PolymerChain::get_block_start()
{
    return block_start;
}