#include <tuple>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "PolymerChain.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(langevinfts, m)
{
    py::class_<ComputationBox>(m, "ComputationBox")
        .def(py::init<std::vector<int>, std::vector<double>>())
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
        .def("integral", overload_cast_<py::array_t<double>>()(&ComputationBox::integral))
        .def("inner_product", overload_cast_<py::array_t<double>,py::array_t<double>>()(&ComputationBox::inner_product))
        .def("multi_inner_product", overload_cast_<int,py::array_t<double>,py::array_t<double>>()(&ComputationBox::multi_inner_product))
        .def("zero_mean", overload_cast_<py::array_t<double>>()(&ComputationBox::zero_mean));

    py::class_<PolymerChain>(m, "PolymerChain")
        .def(py::init<double, std::map<std::string, double>, double,
            std::vector<std::string>, std::vector<double>,
            std::vector<int>, std::vector<int>, std::map<int, std::string>>())
        .def("get_alpha", &PolymerChain::get_alpha)
        .def("get_volume_fraction", &PolymerChain::get_volume_fraction)
        .def("get_n_blocks", &PolymerChain::get_n_blocks)
        .def("get_blocks", &PolymerChain::get_blocks)
        .def("get_block", &PolymerChain::get_block)
        //.def("get_block_species", &PolymerChain::get_block_species)
        .def("get_n_segment_total", &PolymerChain::get_n_segment_total)
        .def("get_n_segment", &PolymerChain::get_n_segment)
        .def("get_block_index_from_edge", &PolymerChain::get_block_index_from_edge)
        .def("get_adjacent_nodes", &PolymerChain::get_adjacent_nodes)
        .def("get_block_indexes", &PolymerChain::get_block_indexes)
        //.def("set_propagator_key", &PolymerChain::set_propagator_key)
        .def("get_propagator_key", &PolymerChain::get_propagator_key);

    py::class_<Mixture>(m, "Mixture")
        .def(py::init<std::string, double, std::map<std::string, double>, bool>())
        .def("get_model_name", &Mixture::get_model_name)
        .def("get_ds", &Mixture::get_ds)
        .def("get_bond_lengths", &Mixture::get_bond_lengths)
        .def("get_n_polymers", &Mixture::get_n_polymers)
        .def("add_polymer", overload_cast_<
            double, std::vector<std::string>, std::vector<double>, std::vector<int>, std::vector<int>, std::map<int, std::string>
            >()(&Mixture::add_polymer))
        .def("add_polymer", overload_cast_<
            double, std::vector<std::string>, std::vector<double>, std::vector<int>, std::vector<int>
            >()(&Mixture::add_polymer))
        .def("get_polymer", &Mixture::get_polymer)
        .def("get_deps_from_key", &Mixture::get_deps_from_key)
        .def("get_monomer_type_from_key", &Mixture::get_monomer_type_from_key)
        .def("get_essential_propagator_codes", &Mixture::get_essential_propagator_codes)
        .def("get_essential_propagator_code", &Mixture::get_essential_propagator_code)
        .def("get_essential_blocks", &Mixture::get_essential_blocks)
        .def("get_essential_block", &Mixture::get_essential_block)
        .def("display_propagators", &Mixture::display_propagators)
        .def("display_blocks", &Mixture::display_blocks);

    py::class_<Pseudo>(m, "Pseudo")
        .def("update_bond_function", &Pseudo::update_bond_function)
        .def("compute_statistics", overload_cast_<
            std::map<std::string,py::array_t<double>>,
            std::map<std::string,py::array_t<double>>>
            ()(&Pseudo::compute_statistics_pybind11), py::return_value_policy::move)
        .def("compute_statistics", overload_cast_<
            std::map<std::string,py::array_t<double>>>
            ()(&Pseudo::compute_statistics_pybind11), py::return_value_policy::move)
        .def("get_monomer_concentration", overload_cast_<std::string>
            ()(&Pseudo::get_monomer_concentration), py::return_value_policy::move)
        .def("get_polymer_concentration", overload_cast_<int>
            ()(&Pseudo::get_polymer_concentration), py::return_value_policy::move)
        .def("get_total_partition", &Pseudo::get_total_partition)
        .def("get_chain_propagator", overload_cast_<int, int, int, int>
            ()(&Pseudo::get_chain_propagator), py::return_value_policy::move)
        .def("compute_stress", &Pseudo::compute_stress);

    py::class_<AndersonMixing>(m, "AndersonMixing")
        .def("reset_count", &AndersonMixing::reset_count)
        .def("calculate_new_fields",overload_cast_<py::array_t<double>,
            py::array_t<double>, double, double>()(&AndersonMixing::calculate_new_fields));

    py::class_<AbstractFactory>(m, "AbstractFactory")
        .def("create_computation_box", &AbstractFactory::create_computation_box)
        .def("create_mixture", &AbstractFactory::create_mixture)
        .def("create_pseudo", &AbstractFactory::create_pseudo)
        .def("create_anderson_mixing", &AbstractFactory::create_anderson_mixing)
        .def("display_info", &AbstractFactory::display_info)
        .def("get_model_name", &AbstractFactory::get_model_name);

    py::class_<PlatformSelector>(m, "PlatformSelector")
        .def(py::init<>())
        .def("avail_platforms", &PlatformSelector::avail_platforms)
        .def("create_factory", overload_cast_<std::string>()(&PlatformSelector::create_factory))
        .def("create_factory", overload_cast_<std::string, std::string>()(&PlatformSelector::create_factory));
}