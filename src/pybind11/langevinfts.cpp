#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "PolymerChain.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "SingleChainStatistics.h"
#include "PlatformSelector.h"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(langevinfts, m)
{
    py::class_<PolymerChain>(m, "PolymerChain")
        .def(py::init<std::vector<int>, std::vector<double>, std::string>())
        .def("get_n_block", &PolymerChain::get_n_block)
        .def("get_n_segment", overload_cast_<>()(&PolymerChain::get_n_segment))
        .def("get_n_segment", overload_cast_<int>()(&PolymerChain::get_n_segment))
        .def("get_n_segment_total", &PolymerChain::get_n_segment_total)
        .def("get_ds", &PolymerChain::get_ds)
        .def("get_relative_length", &PolymerChain::get_relative_length)
        .def("get_bond_length", overload_cast_<>()(&PolymerChain::get_bond_length))
        .def("get_bond_length", overload_cast_<int>()(&PolymerChain::get_bond_length))
        .def("get_model_name", &PolymerChain::get_model_name);

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

    py::class_<Pseudo>(m, "Pseudo")
        .def("update", &Pseudo::update)
        .def("find_phi", overload_cast_<py::array_t<double>, py::array_t<double>,
            py::array_t<double>>()(&Pseudo::find_phi), py::return_value_policy::move)
        .def("get_partition", overload_cast_<int, int>()(&Pseudo::get_partition), py::return_value_policy::move)
        .def("dq_dl", &Pseudo::dq_dl);

    py::class_<AndersonMixing>(m, "AndersonMixing")
        .def("reset_count", &AndersonMixing::reset_count)
        .def("caculate_new_fields",overload_cast_<py::array_t<double>, py::array_t<double>,
            py::array_t<double>, double, double>()(&AndersonMixing::caculate_new_fields));

    py::class_<AbstractFactory>(m, "AbstractFactory")
        .def("create_polymer_chain", &AbstractFactory::create_polymer_chain)
        .def("create_computation_box", &AbstractFactory::create_computation_box)
        .def("create_pseudo", &AbstractFactory::create_pseudo)
        .def("create_anderson_mixing", &AbstractFactory::create_anderson_mixing)
        .def("display_info", &AbstractFactory::display_info);

    py::class_<SingleChainStatistics>(m, "SingleChainStatistics")
        .def(py::init<std::string, std::string>())
        .def("get_model_name", &SingleChainStatistics::get_model_name)
        .def("create_polymer_chain", &SingleChainStatistics::create_polymer_chain)
        .def("create_computation", &SingleChainStatistics::create_computation)
        .def("create_computation_box", &SingleChainStatistics::create_computation_box)
        .def("create_pseudo", &SingleChainStatistics::create_pseudo)
        .def("create_anderson_mixing", &SingleChainStatistics::create_anderson_mixing)
        .def("display_info", &SingleChainStatistics::display_info);

    py::class_<PlatformSelector>(m, "PlatformSelector")
        .def(py::init<>())
        .def("avail_platforms", &PlatformSelector::avail_platforms)
        .def("create_factory", overload_cast_<std::string>()(&PlatformSelector::create_factory));
}