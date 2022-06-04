#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "PolymerChain.h"
#include "SimulationBox.h"
#include "Pseudo.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(langevinfts, m)
{
    py::class_<PolymerChain>(m, "PolymerChain")
        .def(py::init<double, int, double, std::string, double>())
        .def("get_n_contour", &PolymerChain::get_n_contour)
        .def("get_n_contour_a", &PolymerChain::get_n_contour_a)
        .def("get_n_contour_b", &PolymerChain::get_n_contour_b)
        .def("get_f", &PolymerChain::get_f)
        .def("get_ds", &PolymerChain::get_ds)
        .def("get_chi_n", &PolymerChain::get_chi_n)
        .def("get_epsilon", &PolymerChain::get_epsilon)
        .def("get_model_name", &PolymerChain::get_model_name)
        .def("set_chi_n", &PolymerChain::set_chi_n);

    py::class_<SimulationBox>(m, "SimulationBox")
        .def(py::init<std::vector<int>, std::vector<double>>())
        .def("get_dim", &SimulationBox::get_dim)
        .def("get_nx", overload_cast_<>()(&SimulationBox::get_nx))
        .def("get_nx", overload_cast_<int>()(&SimulationBox::get_nx))
        .def("get_lx", overload_cast_<>()(&SimulationBox::get_lx))
        .def("get_lx", overload_cast_<int>()(&SimulationBox::get_lx))
        .def("get_dx", overload_cast_<>()(&SimulationBox::get_dx))
        .def("get_dx", overload_cast_<int>()(&SimulationBox::get_dx))
        .def("get_dv", &SimulationBox::get_dv)
        .def("get_n_grid", &SimulationBox::get_n_grid)
        .def("get_volume", &SimulationBox::get_volume)
        .def("set_lx", &SimulationBox::set_lx)
        .def("integral", overload_cast_<py::array_t<double>>()(&SimulationBox::integral))
        .def("inner_product", overload_cast_<py::array_t<double>,py::array_t<double>>()(&SimulationBox::inner_product))
        .def("multi_inner_product", overload_cast_<int,py::array_t<double>,py::array_t<double>>()(&SimulationBox::multi_inner_product))
        .def("zero_mean", overload_cast_<py::array_t<double>>()(&SimulationBox::zero_mean));

    py::class_<Pseudo>(m, "Pseudo")
        .def("update", &Pseudo::update)
        .def("find_phi", overload_cast_<py::array_t<double>, py::array_t<double>,
            py::array_t<double>, py::array_t<double>>()(&Pseudo::find_phi), py::return_value_policy::move)
        .def("get_partition", overload_cast_<int, int>()(&Pseudo::get_partition), py::return_value_policy::move)
        .def("dq_dl", &Pseudo::dq_dl);

    py::class_<AndersonMixing>(m, "AndersonMixing")
        .def("reset_count", &AndersonMixing::reset_count)
        .def("caculate_new_fields",overload_cast_<py::array_t<double>, py::array_t<double>,
            py::array_t<double>, double, double>()(&AndersonMixing::caculate_new_fields));

    py::class_<AbstractFactory>(m, "AbstractFactory")
        .def("create_polymer_chain", &AbstractFactory::create_polymer_chain)
        .def("create_simulation_box", &AbstractFactory::create_simulation_box)
        .def("create_pseudo", &AbstractFactory::create_pseudo)
        .def("create_anderson_mixing", &AbstractFactory::create_anderson_mixing)
        .def("display_info", &AbstractFactory::display_info);

    py::class_<PlatformSelector>(m, "PlatformSelector")
        .def(py::init<>())
        .def("avail_platforms", &PlatformSelector::avail_platforms)
        .def("create_factory", overload_cast_<std::string>()(&PlatformSelector::create_factory));
}