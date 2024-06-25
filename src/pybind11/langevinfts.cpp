#include <tuple>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "Array.h"
#include "Polymer.h"
#include "PropagatorAnalyzer.h"
#include "ComputationBox.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "Exception.h"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(langevinfts, m)
{
    py::class_<Array>(m, "Array")
        .def("set_data", [](Array& obj, py::array_t<const double> data)
        {
            try
            {
                int size = obj.get_size();
                py::buffer_info buf = data.request();
                if (buf.size != size) {
                    throw_with_line_number("Size of input (" + std::to_string(buf.size) + ") and 'n_grid' (" + std::to_string(size) + ") must match");
                }
                obj.set_data((double*) buf.ptr, size);
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        })
        .def("get_ptr", [](Array& obj)
        {
            return reinterpret_cast<std::uintptr_t>(obj.get_ptr());
        })
        .def("get_size", &Array::get_size);

    py::class_<ComputationBox>(m, "ComputationBox")
        // .def(py::init<std::vector<int>, std::vector<double>>())
        .def("get_dim", &ComputationBox::get_dim)
        .def("get_nx", overload_cast_<>()(&ComputationBox::get_nx))
        .def("get_nx", overload_cast_<int>()(&ComputationBox::get_nx))
        .def("get_lx", overload_cast_<>()(&ComputationBox::get_lx))
        .def("get_lx", overload_cast_<int>()(&ComputationBox::get_lx))
        .def("get_dx", overload_cast_<>()(&ComputationBox::get_dx))
        .def("get_dx", overload_cast_<int>()(&ComputationBox::get_dx))
        .def("get_dv", &ComputationBox::get_dv)
        .def("get_n_grid", &ComputationBox::get_n_grid)
        .def("get_volume", &ComputationBox::get_volume)
        .def("set_lx", &ComputationBox::set_lx)
        .def("integral", [](ComputationBox& obj, py::array_t<double> g)
        {
            const int M = obj.get_n_grid();
            py::buffer_info buf_g = g.request();
            if (buf_g.size != M) {
                throw_with_line_number("Size of input (" + std::to_string(buf_g.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
            }
            return obj.integral((double*) buf_g.ptr);
        })
        .def("inner_product", [](ComputationBox& obj, py::array_t<double> g, py::array_t<double> h)
        {
            const int M = obj.get_n_grid();
            py::buffer_info buf_g = g.request();
            py::buffer_info buf_h = h.request();
            if (buf_g.size != M) {
                throw_with_line_number("Size of input (" + std::to_string(buf_g.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
            }
            if (buf_h.size != M) {
                throw_with_line_number("Size of input (" + std::to_string(buf_h.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
            }
            return obj.inner_product((double*) buf_g.ptr, (double*) buf_h.ptr);
        });
        // .def("multi_inner_product", &ComputationBox::multi_inner_product);
        // .def("zero_mean", overload_cast_<py::array_t<double>>()(&ComputationBox::zero_mean));
        // double multi_inner_product(int n_comp, py::array_t<double> g, py::array_t<double> h) {
        //     py::buffer_info buf1 = g.request();
        //     py::buffer_info buf2 = h.request();
        //     if (buf1.size != n_comp*n_grid) 
        //         throw_with_line_number("Size of input g (" + std::to_string(buf1.size) + ") and 'n_comp x n_grid' (" + std::to_string(n_comp*n_grid) + ") must match");
        //     if (buf2.size != n_comp*n_grid)
        //         throw_with_line_number("Size of input h (" + std::to_string(buf2.size) + ") and 'n_comp x n_grid' (" + std::to_string(n_comp*n_grid) + ") must match");
        //     return multi_inner_product(n_comp, (double*) buf1.ptr, (double*) buf2.ptr);
        // };
        // Void zero_mean(py::array_t<double> g) {
        //     py::buffer_info buf = g.request();
        //     if (buf.size != n_grid) {
        //         throw_with_line_number("Size of input (" + std::to_string(buf.size) + ") and 'n_grid' (" + std::to_string(n_grid) + ") must match");
        //     }
        //     zero_mean((double*) buf.ptr);
        // };

    py::class_<Polymer>(m, "Polymer")
        // .def(py::init<double, std::map<std::string, double>, double,
        //     std::vector<std::string>, std::vector<double>,
        //     std::vector<int>, std::vector<int>, std::map<int, std::string>>())
        .def("get_alpha", &Polymer::get_alpha)
        .def("get_volume_fraction", &Polymer::get_volume_fraction)
        .def("get_n_blocks", &Polymer::get_n_blocks)
        .def("get_blocks", &Polymer::get_blocks)
        .def("get_block", &Polymer::get_block)
        //.def("get_block_species", &Polymer::get_block_species)
        .def("get_n_segment_total", &Polymer::get_n_segment_total)
        .def("get_n_segment", &Polymer::get_n_segment)
        .def("get_block_index_from_edge", &Polymer::get_block_index_from_edge)
        .def("get_adjacent_nodes", &Polymer::get_adjacent_nodes)
        .def("get_block_indexes", &Polymer::get_block_indexes)
        //.def("set_propagator_key", &Polymer::set_propagator_key)
        .def("get_propagator_key", &Polymer::get_propagator_key);

    py::class_<Molecules>(m, "Molecules")
        .def(py::init<std::string, double, std::map<std::string, double>>())
        .def("get_model_name", &Molecules::get_model_name)
        .def("get_ds", &Molecules::get_ds)
        .def("get_bond_lengths", &Molecules::get_bond_lengths)
        .def("get_n_polymer_types", &Molecules::get_n_polymer_types)
        .def("add_polymer", [](Molecules& obj, double volume_fraction, std::vector<std::vector<py::object>> block_list)
        {
            std::vector<BlockInput> block_inputs;
            for (const auto& py_block : block_list) {
                BlockInput block;
                block.monomer_type = py::cast<std::string>(py_block[0]);
                block.contour_length = py::cast<double>(py_block[1]);
                block.v = py::cast<int>(py_block[2]);
                block.u = py::cast<int>(py_block[3]);
                block_inputs.push_back(block);
            }
            obj.add_polymer(volume_fraction, block_inputs, {});
        })
        .def("add_polymer", [](Molecules& obj, double volume_fraction, std::vector<std::vector<py::object>> block_list, std::map<int, std::string> chain_end_to_q_init)
        {
            std::vector<BlockInput> block_inputs;
            for (const auto& py_block : block_list) {
                BlockInput block;
                block.monomer_type = py::cast<std::string>(py_block[0]);
                block.contour_length = py::cast<double>(py_block[1]);
                block.v = py::cast<int>(py_block[2]);
                block.u = py::cast<int>(py_block[3]);
                block_inputs.push_back(block);
            }
            obj.add_polymer(volume_fraction, block_inputs, chain_end_to_q_init);
        })
        .def("get_polymer", &Molecules::get_polymer)
        .def("get_n_solvent_types", &Molecules::get_n_solvent_types)
        .def("add_solvent", &Molecules::add_solvent);
        
    py::class_<PropagatorAnalyzer>(m, "PropagatorAnalyzer")
        .def("get_computation_propagator_codes()", &PropagatorAnalyzer::get_computation_propagator_codes)
        .def("get_computation_propagator", &PropagatorAnalyzer::get_computation_propagator)
        .def("get_computation_blocks", &PropagatorAnalyzer::get_computation_blocks)
        .def("get_computation_block", &PropagatorAnalyzer::get_computation_block)
        .def("display_propagators", &PropagatorAnalyzer::display_propagators)
        .def("display_blocks", &PropagatorAnalyzer::display_blocks)
        .def("get_deps_from_key", &PropagatorCode::get_deps_from_key)
        .def("get_monomer_type_from_key", &PropagatorCode::get_monomer_type_from_key);

    py::class_<PropagatorComputation>(m, "PropagatorComputation")
        .def("update_laplacian_operator", &PropagatorComputation::update_laplacian_operator)
        .def("compute_statistics", [](PropagatorComputation& obj, std::map<std::string,py::array_t<const double>> w_input, py::object q_init)
        {
            try{
                const int M = obj.get_n_grid();
                std::map<std::string, const double*> map_buf_w_input;
                std::map<std::string, const double*> map_buf_q_init;

                //buf_w_input
                for (auto it = w_input.begin(); it != w_input.end(); ++it)
                {
                    py::buffer_info buf_w_input = it->second.request();
                    if (buf_w_input.size != M) {
                        throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_w_input.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
                    }
                    else
                    {
                        map_buf_w_input.insert(std::pair<std::string, const double*>(it->first, (const double*)buf_w_input.ptr));
                    }
                }

                //buf_q_init
                if (!q_init.is_none()) {
                    std::map<std::string, py::array_t<const double>> q_init_map = q_init.cast<std::map<std::string, py::array_t<const double>>>();

                    for (auto it = q_init_map.begin(); it != q_init_map.end(); ++it)
                    {
                        py::buffer_info buf_q_init = it->second.request();
                        if (buf_q_init.size != M) {
                            throw_with_line_number("Size of input q[" + it->first + "] (" + std::to_string(buf_q_init.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
                        }
                        else
                        {
                            map_buf_q_init.insert(std::pair<std::string, const double*>(it->first, (const double*)buf_q_init.ptr));
                        }
                    }
                }

                obj.compute_statistics(map_buf_w_input, map_buf_q_init);
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        }, py::arg("w_input"), py::arg("q_init") = py::none())
        // .def("compute_statistics_device", [](PropagatorComputation& obj, std::map<std::string, const long int> d_w_input, std::map<std::string, const long int> d_q_init)
        // {
        //     try{
        //         std::map<std::string, const double*> map_buf_w_input;
        //         std::map<std::string, const double*> map_buf_q_init;

        //         for (auto it=d_w_input.begin(); it!=d_w_input.end(); ++it)
        //         {
        //             //buf_w_input
        //             const double* w_input_ptr = reinterpret_cast<const double*>(it->second);
        //             map_buf_w_input.insert(std::pair<std::string, const double*>(it->first,(const double *) w_input_ptr));
        //         }

        //         for (auto it=d_q_init.begin(); it!=d_q_init.end(); ++it)
        //         {
        //             //buf_q_init
        //             const double* q_init_ptr = reinterpret_cast<const double*>(it->second);
        //             map_buf_q_init.insert(std::pair<std::string, const double*>(it->first,(const double *) q_init_ptr));
        //         }
        //         obj.compute_statistics_device(map_buf_w_input, map_buf_q_init);
        //     }
        //     catch(std::exception& exc)
        //     {
        //         throw_without_line_number(exc.what());
        //     }
        // })
        .def("get_total_concentration", [](PropagatorComputation& obj, std::string monomer_type)
        {
            try{
                const int M = obj.get_n_grid();
                py::array_t<double> phi = py::array_t<double>(M);
                py::buffer_info buf_phi = phi.request();
                obj.get_total_concentration(monomer_type, (double*) buf_phi.ptr);
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_total_concentration", [](PropagatorComputation& obj, int polymer, std::string monomer_type)
        {
            try{
                const int M = obj.get_n_grid();
                py::array_t<double> phi = py::array_t<double>(M);
                py::buffer_info buf_phi = phi.request();
                obj.get_total_concentration(polymer, monomer_type, (double*) buf_phi.ptr);
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_block_concentration", [](PropagatorComputation& obj, int polymer)
        {
            try{
                const int M = obj.get_n_grid();
                const int N_B = obj.get_n_blocks(polymer);

                py::array_t<double> phi = py::array_t<double>({N_B,M});
                py::buffer_info buf_phi = phi.request();
                obj.get_block_concentration(polymer, (double*) buf_phi.ptr);
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_total_partition", &PropagatorComputation::get_total_partition)
        .def("get_solvent_partition", &PropagatorComputation::get_solvent_partition)
        .def("get_solvent_concentration", [](PropagatorComputation& obj, int s)
        {
            try{
                const int M = obj.get_n_grid();

                py::array_t<double> phi = py::array_t<double>({M});
                py::buffer_info buf_phi = phi.request();
                obj.get_solvent_concentration(s, (double*) buf_phi.ptr);
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_chain_propagator", [](PropagatorComputation& obj, int polymer, int v, int u, int n)
        {
            try{
                const int M = obj.get_n_grid();
                py::array_t<double> q1 = py::array_t<double>(M);
                py::buffer_info buf_q1 = q1.request();
                obj.get_chain_propagator((double*) buf_q1.ptr, polymer, v, u, n);
                return q1;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("compute_stress", &PropagatorComputation::compute_stress)
        .def("check_total_partition", &PropagatorComputation::check_total_partition);

    py::class_<AndersonMixing>(m, "AndersonMixing")
        .def("reset_count", &AndersonMixing::reset_count)
        .def("calculate_new_fields", [](AndersonMixing &obj,
                py::array_t<double> w_current, py::array_t<double> w_deriv,
                double old_error_level, double error_level)
        {
            try{

                int n_var = obj.get_n_var();
                py::array_t<double> w_new = py::array_t<double>(n_var);

                py::buffer_info buf_w_new = w_new.request();
                py::buffer_info buf_w_current = w_current.request();
                py::buffer_info buf_w_deriv = w_deriv.request();

                if (buf_w_new.size != n_var)
                    throw_with_line_number("Size of input w_new (" + std::to_string(buf_w_new.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match");
                if (buf_w_current.size != n_var)
                    throw_with_line_number("Size of input w_current (" + std::to_string(buf_w_current.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match");
                if (buf_w_deriv.size != n_var)
                    throw_with_line_number("Size of input w_deriv (" + std::to_string(buf_w_deriv.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match");

                obj.calculate_new_fields((double *) buf_w_new.ptr, (double *) buf_w_current.ptr, (double *) buf_w_deriv.ptr, old_error_level, error_level);
                return w_new;
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        });

    py::class_<AbstractFactory>(m, "AbstractFactory")
        .def("create_array", overload_cast_<unsigned int>()(&AbstractFactory::create_array))
        // .def("create_computation_box", &AbstractFactory::create_computation_box)
        .def("create_computation_box", [](
            AbstractFactory& obj,
            std::vector<int> nx, std::vector<double> lx,
            py::object bc,
            py::object mask)
        {
            try{
                int M = 1;
                for(size_t d=0; d<nx.size(); d++)
                    M *= nx[d];

                py::buffer_info buf_mask;

                // Check if bc is not None
                std::vector<std::string> bc_vec;
                if (!bc.is_none())
                {
                    py::list bc_list = py::cast<py::list>(bc);
                    for (size_t i = 0; i < py::len(bc_list); ++i)
                        bc_vec.push_back(py::cast<std::string>(bc_list[i]));
                }

                // Check if mask is not None
                if (!mask.is_none())
                {
                    py::array_t<const double> mask_cast = mask.cast<py::array_t<const double>>();
                    buf_mask = mask_cast.request();
                    if (buf_mask.size != M) {
                        throw_with_line_number("Size of input (" + std::to_string(buf_mask.size) + ") and 'n_grid' (" + std::to_string(M) + ") must match");
                    }
                }
                return obj.create_computation_box(nx, lx, bc_vec, (const double *) buf_mask.ptr);
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        }, py::arg("nx"), py::arg("lx"), py::arg("bc") = py::none(), py::arg("mask") = py::none())
        .def("create_molecules_information", &AbstractFactory::create_molecules_information)
        .def("create_propagator_analyzer", &AbstractFactory::create_propagator_analyzer)
        .def("create_pseudospectral_solver", &AbstractFactory::create_pseudospectral_solver)
        .def("create_realspace_solver", &AbstractFactory::create_realspace_solver)
        .def("create_anderson_mixing", &AbstractFactory::create_anderson_mixing)
        .def("display_info", &AbstractFactory::display_info)
        .def("get_model_name", &AbstractFactory::get_model_name);

    py::class_<PlatformSelector>(m, "PlatformSelector")
        .def(py::init<>())
        .def("avail_platforms", &PlatformSelector::avail_platforms)
        .def("create_factory", overload_cast_<std::string>()(&PlatformSelector::create_factory))
        .def("create_factory", overload_cast_<std::string, bool>()(&PlatformSelector::create_factory));
}