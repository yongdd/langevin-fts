/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <iostream>
#include <cassert>
#include <cstdio>
#include <tuple>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ComputationBox.h"
#include "Mixture.h"
#include "PolymerChain.h"
#include "Exception.h"

namespace py = pybind11;

class Pseudo
{
protected:
    ComputationBox *cb;
    Mixture *mx;
    int n_complex_grid;

    void get_boltz_bond(double *boltz_bond, double bond_length_variance,
        std::array<int,3> nx, std::array<double,3> dx, double ds);

    void get_weighted_fourier_basis(
        double *fourier_basis_x, double *fourier_basis_y, double *fourier_basis_z,
        std::array<int,3> nx, std::array<double,3> dx);
public:
    Pseudo(ComputationBox *cb, Mixture *mx);
    virtual ~Pseudo() {};

    virtual void update() = 0;
    virtual void compute_statistics(
        std::map<std::string, double*> q_init,
        std::map<std::string, double*> w_block) = 0;
    virtual double get_total_partition(int polymer) = 0;
    virtual void get_species_concentration(std::string species, double *phi) = 0;
    virtual void get_polymer_concentration(int polymer, double *phi) = 0;
    virtual std::array<double,3> compute_stress() = 0;
    virtual void get_partial_partition(double *q_out, int polymer, int v, int u, int n) = 0;

    // Methods for pybind11
    // void compute_statistics(std::map<std::string,py::array_t<double>> q_init, std::map<std::string,py::array_t<double>> w_block)
    void compute_statistics(std::map<std::string,py::array_t<double>> w_block)
    {
        try{
            const int M = cb->get_n_grid();
            //std::map<std::string,double*> map_buf_q_init;
            std::map<std::string,double*> map_buf_w_block;

            // for (auto it=q_init.begin(); it!=q_init.end(); ++it)
            // {
            //     //buf_q_init
            //     py::buffer_info buf_q_init = it->second.request();
            //     if (buf_q_init.shape[0] != M){
            //         throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_q_init.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
            //     }
            //     else
            //     {
            //         map_buf_q_init.insert(std::pair<std::string,double*>(it->first,(double *)buf_q_init.ptr));
            //     }
            // }

            for (auto it=w_block.begin(); it!=w_block.end(); ++it)
            {
                //buf_w_block
                py::buffer_info buf_w_block = it->second.request();
                if (buf_w_block.shape[0] != M){
                    throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_w_block.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
                }
                else
                {
                    map_buf_w_block.insert(std::pair<std::string,double*>(it->first,(double *)buf_w_block.ptr));
                }
            }
            compute_statistics({}, map_buf_w_block);
        }
        catch(std::exception& exc)
        {
            throw_without_line_number(exc.what());
        }
    }
    py::array_t<double> get_species_concentration(std::string species)
    {
        try{
            const int M = cb->get_n_grid();
            py::array_t<double> phi = py::array_t<double>(M);
            py::buffer_info buf_phi = phi.request();
            get_species_concentration(species, (double*) buf_phi.ptr);
            return std::move(phi);
        }
        catch(std::exception& exc)
        {
            throw_with_line_number(exc.what());
        }
    };

    py::array_t<double> get_polymer_concentration(int polymer)
    {
        try{
            PolymerChain& pc = mx->get_polymer(polymer);
            const int M = cb->get_n_grid();
            const int N_B = pc.get_n_blocks();

            py::array_t<double> phi = py::array_t<double>({N_B,M});
            py::buffer_info buf_phi = phi.request();
            get_polymer_concentration(polymer, (double*) buf_phi.ptr);
            return std::move(phi);
        }
        catch(std::exception& exc)
        {
            throw_with_line_number(exc.what());
        }
    };
    py::array_t<double> get_partial_partition(int polymer, int v, int u, int n)
    {
        try{
            const int M = cb->get_n_grid();
            py::array_t<double> q1 = py::array_t<double>(M);
            py::buffer_info buf_q1 = q1.request();
            get_partial_partition((double*) buf_q1.ptr, polymer, v, u, n);
            return std::move(q1);
        }
        catch(std::exception& exc)
        {
            throw_with_line_number(exc.what());
        }
    };
};
#endif
