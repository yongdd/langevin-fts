/**
 * @file polymerfts_core.cpp
 * @brief Python bindings for the polymer field theory simulation library.
 *
 * This file uses pybind11 to expose C++ classes and functions to Python,
 * creating the `_core` module that is imported by the Python layer.
 *
 * **Exposed Classes:**
 *
 * - ComputationBox (Real/Complex): Grid and FFT operations
 * - Molecules: Polymer architecture definitions
 * - Polymer: Individual polymer chain data
 * - PropagatorComputation (Real/Complex): Propagator solving
 * - AndersonMixing (Real/Complex): Field mixing acceleration
 * - AbstractFactory (Real/Complex): Platform-specific object creation
 * - PlatformSelector: Platform detection and factory creation
 * - PropagatorComputationOptimizer: Propagator computation planning
 *
 * **Type Support:**
 *
 * Classes are templated for both real (double) and complex (std::complex<double>)
 * types. Python class names use suffixes: _Real, _Complex.
 *
 * **NumPy Integration:**
 *
 * All array parameters use py::array_t for zero-copy NumPy interoperability.
 * Size validation is performed to ensure array dimensions match grid size.
 *
 * **Usage from Python:**
 *
 * @code{.py}
 * from langevinfts import _core
 *
 * # Create platform factory
 * factory = _core.PlatformSelector.create_factory("cuda", reduce_memory_usage=False)
 *
 * # Create computation objects
 * cb = factory.create_computation_box(nx=[32,32,32], lx=[4.0,4.0,4.0])
 * molecules = factory.create_molecules_information("continuous", 0.1, {"A": 1.0, "B": 1.0})
 * @endcode
 *
 * @see scft.py for high-level SCFT interface
 * @see lfts.py for Langevin dynamics interface
 */

#include <tuple>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "Array.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "ComputationBox.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "Exception.h"

namespace py = pybind11;

/**
 * @brief Helper for method overload resolution in pybind11.
 *
 * Used to disambiguate overloaded C++ methods when binding to Python.
 */
template <typename... Args>
using overload_cast_ = py::detail::overload_cast_impl<Args...>;


/**
 * @brief Bind ComputationBox<T> class to Python.
 *
 * Creates Python class "ComputationBox{Real|Complex}" with methods:
 * - get_dim(): Grid dimensionality (1, 2, or 3)
 * - get_nx(): Grid points per dimension
 * - get_lx(): Box lengths
 * - get_dx(): Grid spacing
 * - get_dv(): Volume element
 * - get_total_grid(): Total grid points
 * - get_volume(): Total box volume
 * - set_lx(): Update box dimensions
 * - set_lattice_parameters(): Update box dimensions and angles
 * - integral(): Volume integral of field
 * - inner_product(): Inner product of two fields
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module to add class to
 * @param type_name Suffix for class name ("Real" or "Complex")
 */
template<typename T>
void bind_computation_box(py::module &m, const std::string &type_name) {
    std::string class_name = "ComputationBox" + type_name;
    
    py::class_<ComputationBox<T>>(m, class_name.c_str())
        // .def(py::init<std::vector<int>, std::vector<double>>())
        .def("get_dim", &ComputationBox<T>::get_dim)
        .def("get_nx", py::overload_cast<>(&ComputationBox<T>::get_nx, py::const_))
        .def("get_nx", py::overload_cast<int>(&ComputationBox<T>::get_nx, py::const_))
        .def("get_lx", py::overload_cast<>(&ComputationBox<T>::get_lx, py::const_))
        .def("get_lx", py::overload_cast<int>(&ComputationBox<T>::get_lx, py::const_))
        .def("get_dx", py::overload_cast<>(&ComputationBox<T>::get_dx, py::const_))
        .def("get_dx", py::overload_cast<int>(&ComputationBox<T>::get_dx, py::const_))
        .def("get_dv", &ComputationBox<T>::get_dv)
        .def("get_total_grid", &ComputationBox<T>::get_total_grid)
        .def("get_volume", &ComputationBox<T>::get_volume)
        .def("get_angles", &ComputationBox<T>::get_angles)
        .def("get_angles_degrees", &ComputationBox<T>::get_angles_degrees)
        .def("is_orthogonal", &ComputationBox<T>::is_orthogonal)
        .def("get_recip_metric", &ComputationBox<T>::get_recip_metric)
        .def("set_lx", &ComputationBox<T>::set_lx)
        .def("set_lattice_parameters", overload_cast_<std::vector<double>, std::vector<double>>()(&ComputationBox<T>::set_lattice_parameters))
        .def("integral", [](ComputationBox<T>& obj, py::array_t<T> g)
        {
            const int M = obj.get_total_grid();
            py::buffer_info buf_g = g.request();
            if (buf_g.size != M) {
                throw_with_line_number("Size of input (" + std::to_string(buf_g.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
            }
            return obj.integral(static_cast<T*>(buf_g.ptr));
        })
        .def("inner_product", [](ComputationBox<T>& obj, py::array_t<T> g, py::array_t<T> h)
        {
            const int M = obj.get_total_grid();
            py::buffer_info buf_g = g.request();
            py::buffer_info buf_h = h.request();
            if (buf_g.size != M) {
                throw_with_line_number("Size of input (" + std::to_string(buf_g.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
            }
            if (buf_h.size != M) {
                throw_with_line_number("Size of input (" + std::to_string(buf_h.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
            }
            return obj.inner_product(static_cast<T*>(buf_g.ptr), static_cast<T*>(buf_h.ptr));
        });
        // .def("multi_inner_product", &ComputationBox<T>::multi_inner_product);
        // .def("zero_mean", overload_cast_<py::array_t<double>>()(&ComputationBox<T>::zero_mean));
        // double multi_inner_product(int n_comp, py::array_t<double> g, py::array_t<double> h) {
        //     py::buffer_info buf1 = g.request();
        //     py::buffer_info buf2 = h.request();
        //     if (buf1.size != n_comp*total_grid) 
        //         throw_with_line_number("Size of input g (" + std::to_string(buf1.size) + ") and 'n_comp x total_grid' (" + std::to_string(n_comp*total_grid) + ") must match.");
        //     if (buf2.size != n_comp*total_grid)
        //         throw_with_line_number("Size of input h (" + std::to_string(buf2.size) + ") and 'n_comp x total_grid' (" + std::to_string(n_comp*total_grid) + ") must match.");
        //     return multi_inner_product(n_comp, (double*) buf1.ptr, (double*) buf2.ptr);
        // };
        // Void zero_mean(py::array_t<double> g) {
        //     py::buffer_info buf = g.request();
        //     if (buf.size != total_grid) {
        //         throw_with_line_number("Size of input (" + std::to_string(buf.size) + ") and 'total_grid' (" + std::to_string(total_grid) + ") must match.");
        //     }
        //     zero_mean((double*) buf.ptr);
        // };
}

/**
 * @brief Bind PropagatorComputation<T> class to Python.
 *
 * Creates Python class "PropagatorComputation_{Real|Complex}" with methods:
 * - update_laplacian_operator(): Refresh operators after box change
 * - compute_propagators(w_input, q_init=None): Solve propagator equations
 * - advance_propagator_single_segment(q_in, monomer_type): Single step
 * - compute_concentrations(): Calculate segment densities
 * - compute_statistics(w_input, q_init=None): Full SCFT computation
 * - get_total_concentration(monomer_type): Total density by monomer
 * - get_total_concentration(polymer, monomer_type): Per-polymer density
 * - get_total_concentration_gce(fugacity, polymer, monomer_type): Grand canonical
 * - get_block_concentration(polymer): Block-resolved density
 * - get_total_partition(polymer): Partition function
 * - get_solvent_partition(s): Solvent partition
 * - get_solvent_concentration(s): Solvent density
 * - get_chain_propagator(polymer, v, u, n): Extract propagator at step n
 * - compute_stress(): Calculate stress tensor
 * - get_stress(): Retrieve computed stress
 * - get_stress_gce(): Grand canonical stress
 * - check_total_partition(): Validate partition function
 *
 * All array inputs/outputs use NumPy arrays with automatic size validation.
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_propagator_computation(py::module &m, const std::string &type_name) {
    std::string class_name = "PropagatorComputation_" + type_name;
    
    py::class_<PropagatorComputation<T>>(m, class_name.c_str())
        .def("update_laplacian_operator", &PropagatorComputation<T>::update_laplacian_operator)
        .def("compute_propagators", [](PropagatorComputation<T>& obj, std::map<std::string,py::array_t<const T>> w_input, py::object q_init)
        {
            try{
                const int M = obj.get_total_grid();
                std::map<std::string, const T*> map_buf_w_input;
                std::map<std::string, const T*> map_buf_q_init;

                //buf_w_input
                for (auto it = w_input.begin(); it != w_input.end(); ++it)
                {
                    py::buffer_info buf_w_input = it->second.request();
                    if (buf_w_input.size != M) {
                        throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_w_input.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
                    }
                    else
                    {
                        map_buf_w_input.insert(std::pair<std::string, const T*>(it->first, (const T*)buf_w_input.ptr));
                    }
                }

                //buf_q_init
                if (!q_init.is_none()) {
                    std::map<std::string, py::array_t<const T>> q_init_map = q_init.cast<std::map<std::string, py::array_t<const T>>>();

                    for (auto it = q_init_map.begin(); it != q_init_map.end(); ++it)
                    {
                        py::buffer_info buf_q_init = it->second.request();
                        if (buf_q_init.size != M) {
                            throw_with_line_number("Size of input q[" + it->first + "] (" + std::to_string(buf_q_init.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
                        }
                        else
                        {
                            map_buf_q_init.insert(std::pair<std::string, const T*>(it->first, (const T*)buf_q_init.ptr));
                        }
                    }
                }

                obj.compute_propagators(map_buf_w_input, map_buf_q_init);
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        }, py::arg("w_input"), py::arg("q_init") = py::none())
        .def("advance_propagator_single_segment", [](PropagatorComputation<T>& obj, py::array_t<T> q_in, const std::string monomer_type)
        {
            try{
                // Request input buffer
                py::buffer_info buf_q_in = q_in.request();
                T* q_in_ptr = static_cast<T*>(buf_q_in.ptr);

                // Allocate output buffer
                const int M = obj.get_total_grid();
                py::array_t<T> q_out(M);
                py::buffer_info buf_q_out = q_out.request();
                T* q_out_ptr = static_cast<T*>(buf_q_out.ptr);

                // Call C++ implementation
                if (buf_q_in.size != M)
                {
                    throw_with_line_number("Size of input q_in (" + std::to_string(buf_q_in.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
                }
                else
                {
                    obj.advance_propagator_single_segment(q_in_ptr, q_out_ptr, monomer_type);
                }
                return q_out;

            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        })
        .def("compute_concentrations", &PropagatorComputation<T>::compute_concentrations)
        .def("compute_statistics", [](PropagatorComputation<T>& obj, std::map<std::string,py::array_t<const T>> w_input, py::object q_init)
        {
            try{
                const int M = obj.get_total_grid();
                std::map<std::string, const T*> map_buf_w_input;
                std::map<std::string, const T*> map_buf_q_init;

                //buf_w_input
                for (auto it = w_input.begin(); it != w_input.end(); ++it)
                {
                    py::buffer_info buf_w_input = it->second.request();
                    if (buf_w_input.size != M) {
                        throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_w_input.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
                    }
                    else
                    {
                        map_buf_w_input.insert(std::pair<std::string, const T*>(it->first, (const T*)buf_w_input.ptr));
                    }
                }

                //buf_q_init
                if (!q_init.is_none()) {
                    std::map<std::string, py::array_t<const T>> q_init_map = q_init.cast<std::map<std::string, py::array_t<const T>>>();

                    for (auto it = q_init_map.begin(); it != q_init_map.end(); ++it)
                    {
                        py::buffer_info buf_q_init = it->second.request();
                        if (buf_q_init.size != M) {
                            throw_with_line_number("Size of input q[" + it->first + "] (" + std::to_string(buf_q_init.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
                        }
                        else
                        {
                            map_buf_q_init.insert(std::pair<std::string, const T*>(it->first, (const T*)buf_q_init.ptr));
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
        // .def("compute_statistics_device", [](PropagatorComputation<T>& obj, std::map<std::string, const long int> d_w_input, std::map<std::string, const long int> d_q_init)
        // {
        //     try{
        //         std::map<std::string, const T*> map_buf_w_input;
        //         std::map<std::string, const T*> map_buf_q_init;

        //         for (auto it=d_w_input.begin(); it!=d_w_input.end(); ++it)
        //         {
        //             //buf_w_input
        //             const T* w_input_ptr = reinterpret_cast<const T*>(it->second);
        //             map_buf_w_input.insert(std::pair<std::string, const T*>(it->first,(const T *) w_input_ptr));
        //         }

        //         for (auto it=d_q_init.begin(); it!=d_q_init.end(); ++it)
        //         {
        //             //buf_q_init
        //             const T* q_init_ptr = reinterpret_cast<const T*>(it->second);
        //             map_buf_q_init.insert(std::pair<std::string, const T*>(it->first,(const T *) q_init_ptr));
        //         }
        //         obj.compute_statistics_device(map_buf_w_input, map_buf_q_init);
        //     }
        //     catch(std::exception& exc)
        //     {
        //         throw_without_line_number(exc.what());
        //     }
        // })
        .def("get_total_concentration", [](PropagatorComputation<T>& obj, std::string monomer_type)
        {
            try{
                const int M = obj.get_total_grid();
                py::array_t<T> phi = py::array_t<T>(M);
                py::buffer_info buf_phi = phi.request();
                obj.get_total_concentration(monomer_type, static_cast<T*>(buf_phi.ptr));
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_total_concentration", [](PropagatorComputation<T>& obj, int polymer, std::string monomer_type)
        {
            try{
                const int M = obj.get_total_grid();
                py::array_t<T> phi = py::array_t<T>(M);
                py::buffer_info buf_phi = phi.request();
                obj.get_total_concentration(polymer, monomer_type, static_cast<T*>(buf_phi.ptr));
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_total_concentration_gce", [](PropagatorComputation<T>& obj, double fugacity, int polymer, std::string monomer_type)
        {
            try{
                const int M = obj.get_total_grid();
                py::array_t<T> phi = py::array_t<T>(M);
                py::buffer_info buf_phi = phi.request();
                obj.get_total_concentration_gce(fugacity, polymer, monomer_type, static_cast<T*>(buf_phi.ptr));
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_block_concentration", [](PropagatorComputation<T>& obj, int polymer)
        {
            try{
                const int M = obj.get_total_grid();
                const int N_B = obj.get_n_blocks(polymer);

                py::array_t<T> phi = py::array_t<T>({N_B,M});
                py::buffer_info buf_phi = phi.request();
                obj.get_block_concentration(polymer, static_cast<T*>(buf_phi.ptr));
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_total_partition", &PropagatorComputation<T>::get_total_partition)
        .def("get_solvent_partition", &PropagatorComputation<T>::get_solvent_partition)
        .def("get_solvent_concentration", [](PropagatorComputation<T>& obj, int s)
        {
            try{
                const int M = obj.get_total_grid();

                py::array_t<T> phi = py::array_t<T>({M});
                py::buffer_info buf_phi = phi.request();
                obj.get_solvent_concentration(s, static_cast<T*>(buf_phi.ptr));
                return phi;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("get_chain_propagator", [](PropagatorComputation<T>& obj, int polymer, int v, int u, int n)
        {
            try{
                const int M = obj.get_total_grid();
                py::array_t<T> q1 = py::array_t<T>(M);
                py::buffer_info buf_q1 = q1.request();
                obj.get_chain_propagator(static_cast<T*>(buf_q1.ptr), polymer, v, u, n);
                return q1;
            }
            catch(std::exception& exc)
            {
                throw_with_line_number(exc.what());
            }
        })
        .def("compute_stress", &PropagatorComputation<T>::compute_stress)
        .def("get_stress", &PropagatorComputation<T>::get_stress)
        .def("get_stress_gce", &PropagatorComputation<T>::get_stress_gce)
        .def("check_total_partition", &PropagatorComputation<T>::check_total_partition)
        .def("set_sdc_imex_mode", &PropagatorComputation<T>::set_sdc_imex_mode,
             "Enable/disable IMEX mode for SDC solver. "
             "IMEX treats diffusion implicitly and reaction explicitly, faster for periodic BC in 2D/3D.",
             py::arg("enabled"))
        .def("set_cell_averaged_bond", &PropagatorComputation<T>::set_cell_averaged_bond,
             "Enable/disable cell-averaged bond function. "
             "Cell-averaging applies sinc filtering to ensure non-negativity of bond function in real space.",
             py::arg("enabled"));
}

/**
 * @brief Bind AndersonMixing<T> class to Python.
 *
 * Creates Python class "AndersonMixing_{Real|Complex}" with methods:
 * - reset_count(): Clear history and restart mixing
 * - calculate_new_fields(w_current, w_deriv, old_error, error): Compute mixed field
 *
 * The calculate_new_fields method takes NumPy arrays for:
 * - w_current: Current field values
 * - w_deriv: Field residual (w_new - w_current from SCFT equations)
 * - old_error_level, error_level: Error norms for adaptive mixing
 *
 * Returns: w_new NumPy array with the mixed field
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_anderson_mixing(py::module &m, const std::string &type_name) {
    std::string class_name = "AndersonMixing_" + type_name;
    
    py::class_<AndersonMixing<T>>(m, class_name.c_str())
        .def("reset_count", &AndersonMixing<T>::reset_count)
        .def("calculate_new_fields", [](AndersonMixing<T> &obj,
                py::array_t<T> w_current, py::array_t<T> w_deriv,
                double old_error_level, double error_level)
        {
            try{
                int n_var = obj.get_n_var();
                py::array_t<T> w_new = py::array_t<T>(n_var);

                py::buffer_info buf_w_new = w_new.request();
                py::buffer_info buf_w_current = w_current.request();
                py::buffer_info buf_w_deriv = w_deriv.request();

                if (buf_w_new.size != n_var)
                    throw_with_line_number("Size of input w_new (" + std::to_string(buf_w_new.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match.");
                if (buf_w_current.size != n_var)
                    throw_with_line_number("Size of input w_current (" + std::to_string(buf_w_current.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match.");
                if (buf_w_deriv.size != n_var)
                    throw_with_line_number("Size of input w_deriv (" + std::to_string(buf_w_deriv.size) + ") and 'n_var' (" + std::to_string(n_var) + ") must match.");

                obj.calculate_new_fields(static_cast<T*>(buf_w_new.ptr), 
                                         static_cast<T*>(buf_w_current.ptr),
                                         static_cast<T*>(buf_w_deriv.ptr), old_error_level, error_level);
                return w_new;
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        });
}

/**
 * @brief Bind AbstractFactory<T> class to Python.
 *
 * Creates Python class "AbstractFactory_{Real|Complex}" with factory methods:
 * - create_computation_box(nx, lx, bc=None, mask=None): Create ComputationBox
 * - create_molecules_information(): Create Molecules container
 * - create_propagator_computation_optimizer(): Create optimizer
 * - create_propagator_computation(): Create propagator solver
 * - create_anderson_mixing(): Create Anderson mixing
 * - display_info(): Print platform information
 *
 * The factory pattern ensures all created objects are compatible with
 * the selected platform (CPU-MKL or CUDA).
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_abstract_factory(py::module &m, const std::string &type_name)
{
    std::string class_name = "AbstractFactory_" + type_name;
    
    py::class_<AbstractFactory<T>>(m, class_name.c_str())
        // .def("create_array", overload_cast_<unsigned int>()(&AbstractFactory<double>::create_array))
        // .def("create_computation_box", &AbstractFactory::create_computation_box)
        // .def("get_model_name", &AbstractFactory::get_model_name);
        .def("create_computation_box", [](
            AbstractFactory<T>& obj,
            std::vector<int> nx, std::vector<double> lx,
            py::object angles,
            py::object bc,
            py::object mask)
        {
            try {
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
                    py::array_t<const T> mask_cast = mask.cast<py::array_t<const T>>();
                    buf_mask = mask_cast.request();
                    if (buf_mask.size != M) {
                        throw_with_line_number("Size of input (" + std::to_string(buf_mask.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match");
                    }
                }

                // Check if angles is provided
                if (!angles.is_none())
                {
                    py::list angles_list = py::cast<py::list>(angles);
                    std::vector<double> angles_vec;
                    for (size_t i = 0; i < py::len(angles_list); ++i)
                        angles_vec.push_back(py::cast<double>(angles_list[i]));
                    return obj.create_computation_box(nx, lx, bc_vec, angles_vec, static_cast<double*>(buf_mask.ptr));
                }
                else
                {
                    return obj.create_computation_box(nx, lx, bc_vec, static_cast<double*>(buf_mask.ptr));
                }
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        }, py::arg("nx"), py::arg("lx"), py::arg("angles") = py::none(), py::arg("bc") = py::none(), py::arg("mask") = py::none())
        .def("create_molecules_information", &AbstractFactory<T>::create_molecules_information)
        .def("create_propagator_computation_optimizer", &AbstractFactory<T>::create_propagator_computation_optimizer)
        .def("create_propagator_computation", &AbstractFactory<T>::create_propagator_computation)
        .def("create_anderson_mixing", &AbstractFactory<T>::create_anderson_mixing)
        .def("display_info", &AbstractFactory<T>::display_info);
}

/**
 * @brief Main pybind11 module definition.
 *
 * Creates the `_core` Python module containing all bindings.
 *
 * **Module Contents:**
 *
 * Classes (non-templated):
 * - Polymer: Single polymer chain with topology
 * - Molecules: Container for all polymers and solvents
 * - PropagatorComputationOptimizer: Optimized propagator scheduling
 * - PlatformSelector: Platform detection and factory creation
 *
 * Template classes (Real and Complex variants):
 * - ComputationBoxReal, ComputationBoxComplex
 * - PropagatorComputation_Real, PropagatorComputation_Complex
 * - AndersonMixing_Real, AndersonMixing_Complex
 * - AbstractFactory_Real, AbstractFactory_Complex
 *
 * **Platform Selection:**
 *
 * Use PlatformSelector.create_factory() to get the appropriate factory:
 * @code{.py}
 * factory = _core.PlatformSelector.create_factory("cuda", type="real")
 * factory = _core.PlatformSelector.create_factory("cpu-mkl", type="complex")
 * @endcode
 *
 * The factory then creates all necessary platform-specific objects.
 */
PYBIND11_MODULE(_core, m)
{
    // py::class_<Array>(m, "Array")
    //     .def("set_data", [](Array& obj, py::array_t<const double> data)
    //     {
    //         try
    //         {
    //             int size = obj.get_size();
    //             py::buffer_info buf = data.request();
    //             if (buf.size != size) {
    //                 throw_with_line_number("Size of input (" + std::to_string(buf.size) + ") and 'total_grid' (" + std::to_string(size) + ") must match");
    //             }
    //             obj.set_data((double*) buf.ptr, size);
    //         }
    //         catch(std::exception& exc)
    //         {
    //             throw_without_line_number(exc.what());
    //         }
    //     })
    //     .def("get_ptr", [](Array& obj)
    //     {
    //         return reinterpret_cast<std::uintptr_t>(obj.get_ptr());
    //     })
    //     .def("get_size", &Array::get_size);


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
        .def("get_propagator_key", &Polymer::get_propagator_key)
        .def("display_architecture", &Polymer::display_architecture);

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
        .def("add_solvent", &Molecules::add_solvent)
        .def("display_architectures", &Molecules::display_architectures);
        
    py::class_<PropagatorComputationOptimizer>(m, "PropagatorComputationOptimizer")
        .def("get_computation_propagators()", &PropagatorComputationOptimizer::get_computation_propagators)
        .def("get_computation_propagator", &PropagatorComputationOptimizer::get_computation_propagator)
        .def("get_computation_blocks", &PropagatorComputationOptimizer::get_computation_blocks)
        .def("get_computation_block", &PropagatorComputationOptimizer::get_computation_block)
        .def("display_statistics", &PropagatorComputationOptimizer::display_statistics)
        .def("display_propagators", &PropagatorComputationOptimizer::display_propagators)
        .def("display_blocks", &PropagatorComputationOptimizer::display_blocks)
        .def("get_deps_from_key", &PropagatorCode::get_deps_from_key)
        .def("get_monomer_type_from_key", &PropagatorCode::get_monomer_type_from_key);

    // Bind ComputationBox
    bind_computation_box<double>(m, "Real");
    bind_computation_box<std::complex<double>>(m, "Complex");

    // Bind PropagatorComputation
    bind_propagator_computation<double>(m, "Real");
    bind_propagator_computation<std::complex<double>>(m, "Complex");

    // Bind AndersonMixing
    bind_anderson_mixing<double>(m, "Real");
    bind_anderson_mixing<std::complex<double>>(m, "Complex");

    // Bind AbstractFactory
    bind_abstract_factory<double>(m, "Real");
    bind_abstract_factory<std::complex<double>>(m, "Complex");

    py::class_<PlatformSelector>(m, "PlatformSelector")
        .def(py::init<>())
        .def("avail_platforms", &PlatformSelector::avail_platforms)
        .def_static("create_factory", [](std::string platform_name, bool reduce_memory_usage,
                                         std::string type)
        {
            // Converting type to lowercase
            std::transform(type.begin(), type.end(), type.begin(),
            [](unsigned char c){ return std::tolower(c); });

            if (type == "real")
            {
                return py::cast(PlatformSelector::create_factory_real(platform_name, reduce_memory_usage));
            }
            else if (type == "complex")
            {
                return py::cast(PlatformSelector::create_factory_complex(platform_name, reduce_memory_usage));
            }
            else {
                throw std::runtime_error("Invalid type parameter. Must be either 'real' or 'complex'");
            }
        }, py::arg("platform_name"), py::arg("reduce_memory_usage") = false,
           py::arg("type") = "real");
}