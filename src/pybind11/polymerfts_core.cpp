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
 * from polymerfts import _core
 *
 * # Create platform factory
 * factory = _core.PlatformSelector.create_factory("cuda", reduce_memory=False)
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
#include <cstring>
#include <algorithm>
#include <cctype>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include "GridLimits.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "ComputationBox.h"
#include "PropagatorComputation.h"
#include "AndersonMixing.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"
#include "SpaceGroup.h"
#include "Exception.h"

namespace py = pybind11;

// Input arrays are taken C-contiguous and in the expected dtype: a strided
// NumPy view or a float32 array is copied/cast, never read as if contiguous.
template <typename T>
using array_in = py::array_t<T, py::array::c_style | py::array::forcecast>;

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
 * - get_angles(), get_angles_degrees(), is_orthogonal(), get_recip_metric(): Cell geometry
 * - set_lx(): Update box dimensions
 * - set_lattice_parameters(): Update box dimensions and angles
 * - set_space_group(), get_space_group(), get_n_basis(), get_orbit_counts(): Reduced basis
 * - integral(), mean(): Volume integral / average of a field
 * - inner_product(), inner_product_inverse_weight(), multi_inner_product(): Inner products
 * - zero_mean(): Subtract the mean in place (C-contiguous array required)
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
        // Space group methods
        .def("set_space_group", &ComputationBox<T>::set_space_group, py::keep_alive<1, 2>())
        .def("get_space_group", &ComputationBox<T>::get_space_group, py::return_value_policy::reference_internal)
        .def("get_n_basis", &ComputationBox<T>::get_n_basis)
        .def("get_orbit_counts", &ComputationBox<T>::get_orbit_counts)
        // Field operations: use get_n_basis() for size (n_reduced if space group set, else total_grid)
        .def("integral", [](ComputationBox<T>& obj, array_in<T> g)
        {
            const int N = obj.get_n_basis();
            py::buffer_info buf_g = g.request();
            if (buf_g.size != N) {
                throw_with_line_number("Size of input (" + std::to_string(buf_g.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
            }
            return obj.integral(static_cast<T*>(buf_g.ptr));
        })
        .def("mean", [](ComputationBox<T>& obj, array_in<T> g)
        {
            const int N = obj.get_n_basis();
            py::buffer_info buf_g = g.request();
            if (buf_g.size != N) {
                throw_with_line_number("Size of input (" + std::to_string(buf_g.size) + ") and 'n_basis' (" + std::to_string(N) + ") must match.");
            }
            return obj.mean(static_cast<T*>(buf_g.ptr));
        })
        .def("inner_product", [](ComputationBox<T>& obj, array_in<T> g, array_in<T> h)
        {
            const int N = obj.get_n_basis();
            py::buffer_info buf_g = g.request();
            py::buffer_info buf_h = h.request();
            if (buf_g.size != N) {
                throw_with_line_number("Size of input g (" + std::to_string(buf_g.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
            }
            if (buf_h.size != N) {
                throw_with_line_number("Size of input h (" + std::to_string(buf_h.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
            }
            return obj.inner_product(static_cast<T*>(buf_g.ptr), static_cast<T*>(buf_h.ptr));
        })
        .def("inner_product_inverse_weight", [](ComputationBox<T>& obj, array_in<T> g, array_in<T> h, array_in<T> w)
        {
            const int N = obj.get_n_basis();
            py::buffer_info buf_g = g.request();
            py::buffer_info buf_h = h.request();
            py::buffer_info buf_w = w.request();
            if (buf_g.size != N || buf_h.size != N || buf_w.size != N) {
                throw_with_line_number("All input arrays must have size " + std::to_string(N));
            }
            return obj.inner_product_inverse_weight(static_cast<T*>(buf_g.ptr), static_cast<T*>(buf_h.ptr), static_cast<T*>(buf_w.ptr));
        })
        .def("multi_inner_product", [](ComputationBox<T>& obj, int n_comp, array_in<T> g, array_in<T> h)
        {
            const long long n_total = static_cast<long long>(n_comp) * obj.get_n_basis();
            py::buffer_info buf_g = g.request();
            py::buffer_info buf_h = h.request();
            if (buf_g.size != n_total) {
                throw_with_line_number("Size of g (" + std::to_string(buf_g.size) + ") and n_comp * n_grid (" + std::to_string(n_total) + ") must match.");
            }
            if (buf_h.size != n_total) {
                throw_with_line_number("Size of h (" + std::to_string(buf_h.size) + ") and n_comp * n_grid (" + std::to_string(n_total) + ") must match.");
            }
            return obj.multi_inner_product(n_comp, static_cast<T*>(buf_g.ptr), static_cast<T*>(buf_h.ptr));
        })
        .def("zero_mean", [](ComputationBox<T>& obj, py::array_t<T, 0> g)
        {
            // Modified in place: a converted copy would leave the caller's array
            // unchanged, so the array must already be C-contiguous and of the field type
            // (a wrong dtype is rejected by pybind11 before the call).
            if (!(g.flags() & py::array::c_style))
                throw_with_line_number("zero_mean modifies the array in place; pass a C-contiguous array.");
            const int N = obj.get_n_basis();
            py::buffer_info buf_g = g.request();
            if (buf_g.size != N) {
                throw_with_line_number("Size of input (" + std::to_string(buf_g.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
            }
            obj.zero_mean(static_cast<T*>(buf_g.ptr));
        }, py::arg("g").noconvert());
}

/**
 * @brief Bind PropagatorComputation<T> class to Python.
 *
 * Creates Python class "PropagatorComputation_{Real|Complex}" with methods:
 * - update_laplacian_operator(): Refresh operators after box change
 * - get_cb(): Get ComputationBox (use cb.get_n_basis(), cb.get_total_grid(), etc.)
 * - compute_propagators(w_input, q_init=None): Solve propagator equations
 * - advance_propagator_single_segment(q_in, p, v, u): Single step for block (v,u) of polymer p
 *   (internal; use compute_propagators() and get_chain_propagator() instead)
 * - compute_concentrations(): Calculate segment densities
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
 * - set_force_off_diagonal_stress(), get_force_off_diagonal_stress(): Cross-term stress for orthogonal boxes
 *
 * When space group is set:
 * - compute_propagators() accepts reduced basis w (size: n_reduced)
 * - get_total_concentration() returns reduced basis phi (size: n_reduced)
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
        .def("get_cb", &PropagatorComputation<T>::get_cb, py::return_value_policy::reference_internal)
        // compute_propagators: accepts reduced basis w (n_grid) when space group is set,
        // otherwise accepts full grid w (total_grid)
        .def("compute_propagators", [](PropagatorComputation<T>& obj, std::map<std::string, array_in<const T>> w_input, py::object q_init)
        {
            try{
                const int N = obj.get_cb()->get_n_basis();
                const int M = obj.get_cb()->get_total_grid();
                const bool use_reduced = (obj.get_space_group() != nullptr);

                std::map<std::string, const T*> map_buf_w_input;
                std::map<std::string, const T*> map_buf_q_init;

                // Validate and collect w_input
                for (auto it = w_input.begin(); it != w_input.end(); ++it)
                {
                    py::buffer_info buf_w_input = it->second.request();
                    if (buf_w_input.size != N) {
                        throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_w_input.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
                    }
                    map_buf_w_input[it->first] = (const T*)buf_w_input.ptr;
                }

                // Handle q_init if provided (always full grid)
                if (!q_init.is_none()) {
                    std::map<std::string, array_in<const T>> q_init_map = q_init.cast<std::map<std::string, array_in<const T>>>();

                    for (auto it = q_init_map.begin(); it != q_init_map.end(); ++it)
                    {
                        py::buffer_info buf_q_init = it->second.request();
                        if (buf_q_init.size != M) {
                            throw_with_line_number("Size of input q[" + it->first + "] (" + std::to_string(buf_q_init.size) + ") and 'total_grid' (" + std::to_string(M) + ") must match.");
                        }
                        map_buf_q_init[it->first] = (const T*)buf_q_init.ptr;
                    }
                }

                if (use_reduced) {
                    obj.compute_propagators_reduced(map_buf_w_input, map_buf_q_init);
                } else {
                    obj.compute_propagators(map_buf_w_input, map_buf_q_init);
                }
            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        }, py::arg("w_input"), py::arg("q_init") = py::none())
        .def("advance_propagator_single_segment", [](PropagatorComputation<T>& obj, array_in<T> q_in, int p, int v, int u)
        {
            try{
                // Request input buffer
                py::buffer_info buf_q_in = q_in.request();
                T* q_in_ptr = static_cast<T*>(buf_q_in.ptr);

                // Allocate output buffer
                const int M = obj.get_cb()->get_total_grid();
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
                    obj.advance_propagator_single_segment(q_in_ptr, q_out_ptr, p, v, u);
                }
                return q_out;

            }
            catch(std::exception& exc)
            {
                throw_without_line_number(exc.what());
            }
        })
        .def("compute_concentrations", &PropagatorComputation<T>::compute_concentrations)
        .def("get_total_concentration", [](PropagatorComputation<T>& obj, std::string monomer_type)
        {
            try{
                const int N = obj.get_cb()->get_n_basis();  // n_reduced (with space group) or total_grid
                py::array_t<T> phi(N);
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
                const int N = obj.get_cb()->get_n_basis();  // n_reduced (with space group) or total_grid
                py::array_t<T> phi = py::array_t<T>(N);
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
                const int N = obj.get_cb()->get_n_basis();  // n_reduced (with space group) or total_grid
                py::array_t<T> phi = py::array_t<T>(N);
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
                const int N = obj.get_cb()->get_n_basis();  // n_reduced (with space group) or total_grid
                const int N_B = obj.get_n_blocks(polymer);

                py::array_t<T> phi = py::array_t<T>({N_B,N});
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
                const int N = obj.get_cb()->get_n_basis();  // n_reduced (with space group) or total_grid

                py::array_t<T> phi = py::array_t<T>({N});
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
                const int N = obj.get_cb()->get_n_basis();  // n_reduced (with space group) or total_grid
                py::array_t<T> q1 = py::array_t<T>(N);
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
        .def("set_force_off_diagonal_stress", &PropagatorComputation<T>::set_force_off_diagonal_stress,
            "Force cross-term (V12/V13/V23) stress accumulation even for orthogonal boxes. "
            "Needed when angle optimization starts at exactly 90 degree angles.")
        .def("get_force_off_diagonal_stress", &PropagatorComputation<T>::get_force_off_diagonal_stress)
        .def("get_stress", &PropagatorComputation<T>::get_stress)
        .def("get_stress_gce", &PropagatorComputation<T>::get_stress_gce)
        .def("check_total_partition", &PropagatorComputation<T>::check_total_partition);
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
                array_in<T> w_current, array_in<T> w_deriv,
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
 * the selected platform (CPU-FFTW or CUDA).
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
        .def("create_computation_box", [](
            AbstractFactory<T>& obj,
            std::vector<int> nx, std::vector<double> lx,
            py::object angles,
            py::object bc,
            py::object mask)
        {
            try {
                long long M = 1;   // 64 bits; ComputationBox rejects grids beyond MAX_GRID_POINTS
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
                    array_in<double> mask_cast = mask.cast<array_in<double>>();   // the mask is real for both field types
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
        // keep_alive: the C++ objects hold raw pointers to their arguments, so the
        // Python wrappers of those arguments must outlive the created object
        .def("create_propagator_computation_optimizer", &AbstractFactory<T>::create_propagator_computation_optimizer,
            py::keep_alive<0, 2>())
        .def("create_propagator_computation", &AbstractFactory<T>::create_propagator_computation,
            py::arg("cb"), py::arg("molecules"), py::arg("propagator_computation_optimizer"),
            py::arg("numerical_method"), py::arg("space_group") = nullptr,
            py::keep_alive<0, 2>(), py::keep_alive<0, 3>(), py::keep_alive<0, 4>(), py::keep_alive<0, 6>())
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
 * - SpaceGroup: Space-group symmetry and reduced bases
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
 * factory = _core.PlatformSelector.create_factory("cpu-fftw", type="complex")
 * @endcode
 *
 * The factory then creates all necessary platform-specific objects.
 */
// Field conversions of SpaceGroup. The input holds n_fields fields of n_in values
// each, either flat (1-D) or as (n_fields, n_in); the output keeps the layout of
// the input: a 1-D input gives a 1-D output, a 2-D input an (n_fields, n_out) array.
static int sg_field_count(const py::buffer_info& in, int n_in, const char* n_in_name)
{
    if (in.ndim < 1)
        throw_with_line_number("Input must be at least a 1-D array.");
    py::ssize_t total = 1;
    for (py::ssize_t i = 0; i < in.ndim; ++i)
        total *= in.shape[i];
    const int n_fields = static_cast<int>(total / n_in);
    if (n_fields < 1 || total != static_cast<py::ssize_t>(n_fields) * n_in)
        throw_with_line_number("Field size (" + std::to_string(total) + ") must be a multiple of " +
                               n_in_name + " (" + std::to_string(n_in) + ").");
    return n_fields;
}

static py::array_t<double> sg_output_like(const py::buffer_info& in, int n_fields, int n_out)
{
    if (in.ndim == 1)
        return py::array_t<double>(static_cast<py::ssize_t>(n_fields) * n_out);
    return py::array_t<double>({static_cast<py::ssize_t>(n_fields), static_cast<py::ssize_t>(n_out)});
}

PYBIND11_MODULE(_core, m)
{
    // Grid-size limits of the 32-bit grid indices (GridLimits.h)
    m.attr("MAX_GRID_POINTS") = MAX_GRID_POINTS;
    m.attr("MAX_GRID_ARRAY") = MAX_GRID_ARRAY;


    py::class_<Polymer>(m, "Polymer")
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
        .def("get_global_ds", &Molecules::get_global_ds)
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
        .def("get_computation_propagators", &PropagatorComputationOptimizer::get_computation_propagators)
        .def("get_computation_propagator", &PropagatorComputationOptimizer::get_computation_propagator)
        .def("get_computation_blocks", &PropagatorComputationOptimizer::get_computation_blocks)
        .def("get_computation_block", &PropagatorComputationOptimizer::get_computation_block)
        .def("display_statistics", &PropagatorComputationOptimizer::display_statistics)
        .def("display_propagators", &PropagatorComputationOptimizer::display_propagators)
        .def("display_blocks", &PropagatorComputationOptimizer::display_blocks)
        .def_static("get_deps_from_key", &PropagatorCode::get_deps_from_key)
        .def_static("get_monomer_type_from_key", &PropagatorCode::get_monomer_type_from_key);

    // Bind SpaceGroup
    py::class_<SpaceGroup>(m, "SpaceGroup",
        R"doc(Space group symmetry for reduced basis representation.

Reduces computational cost by exploiting crystallographic symmetry.
Fields can be represented using only reduced-basis points.

Common space groups:
  - Im-3m (229, hall=529): BCC
  - Ia-3d (230, hall=530): Gyroid
  - Fm-3m (225, hall=523): FCC
  - P6_3/mmc (194, hall=488): HCP, PL

Example:
    >>> sg = SpaceGroup([32, 32, 32], "Im-3m", 529)
    >>> print(f"Reduction: {sg.get_total_grid() / sg.get_n_reduced_basis():.1f}x")
)doc")
        .def(py::init<std::vector<int>, int>(), py::arg("nx"), py::arg("hall_number"),
             "Create SpaceGroup from grid dimensions and Hall number.")
        .def(py::init<std::vector<int>, const std::string&, int>(),
             py::arg("nx"), py::arg("symbol"), py::arg("hall_number") = -1,
             "Create SpaceGroup from grid dimensions and ITA symbol.")
        .def_static("hall_numbers_from_symbol", &SpaceGroup::hall_numbers_from_symbol,
             "Return all Hall numbers for an ITA symbol.")
        .def("get_hall_number", &SpaceGroup::get_hall_number,
             "Hall number (1-530).")
        .def("get_spacegroup_number", &SpaceGroup::get_spacegroup_number,
             "International space group number (1-230).")
        .def("get_spacegroup_symbol", &SpaceGroup::get_spacegroup_symbol,
             "ITA short symbol.")
        .def("uses_hexagonal_axes", &SpaceGroup::uses_hexagonal_axes,
             "True for hexagonal/trigonal groups in a hexagonal-axes setting (a = b, gamma = 120); "
             "False for rhombohedral-axes settings (a = b = c, alpha = beta = gamma).")
        .def("get_crystal_system", &SpaceGroup::get_crystal_system,
             "Crystal system name.")
        .def("get_nx", &SpaceGroup::get_nx,
             "Grid dimensions [nx, ny, nz].")
        .def("get_total_grid", &SpaceGroup::get_total_grid,
             "Total number of grid points.")
        .def("get_n_reduced_basis", &SpaceGroup::get_n_reduced_basis,
             "Number of reduced-basis points.")
        .def("get_n_reduced_basis_full", &SpaceGroup::get_n_reduced_basis_full,
             "Number of irreducible-basis points (full space group).")
        .def("get_n_symmetry_ops", &SpaceGroup::get_n_symmetry_ops,
             "Number of symmetry operations.")
        .def("get_reduced_basis_indices", &SpaceGroup::get_reduced_basis_indices,
             "Flat indices of reduced-basis points in full grid.")
        .def("get_full_to_reduced_map", &SpaceGroup::get_full_to_reduced_map,
             "Map from full grid index to reduced-basis index.")
        .def("get_orbit_counts", &SpaceGroup::get_orbit_counts,
             "Number of equivalent points for each reduced-basis point.")
        .def("physical_basis_name", &SpaceGroup::physical_basis_name,
             "Name of the active physical basis: 'irreducible', 'pmmm', 'm3' or 'z-mirror'.")
        .def("in_use", &SpaceGroup::in_use,
             "True once a computation box or solver holds this group; the physical basis is then frozen.")
        .def("enable_pmmm_physical_basis", &SpaceGroup::enable_pmmm_physical_basis,
             "Enable Pmmm physical basis (1/8 grid) mapping.")
        .def("using_pmmm_physical_basis", &SpaceGroup::using_pmmm_physical_basis,
             "Return True if Pmmm physical basis is enabled.")
        .def("m3_crysfft_feasible", &SpaceGroup::m3_crysfft_feasible,
             "True when the recursive 3m CrysFFT, which the m3 physical basis needs, is feasible on this grid.")
        .def("enable_m3_physical_basis", &SpaceGroup::enable_m3_physical_basis,
             "Enable M3 physical basis (1/8 even-index grid) mapping.")
        .def("using_m3_physical_basis", &SpaceGroup::using_m3_physical_basis,
             "Return True if M3 physical basis is enabled.")
        .def("enable_z_mirror_physical_basis", &SpaceGroup::enable_z_mirror_physical_basis,
             "Enable z-mirror physical basis (half grid along z).")
        .def("using_z_mirror_physical_basis", &SpaceGroup::using_z_mirror_physical_basis,
             "Return True if z-mirror physical basis is enabled.")
        .def("to_reduced_basis", [](SpaceGroup& obj, array_in<double> full_field)
        {
            py::buffer_info buf = full_field.request();
            const int n_fields = sg_field_count(buf, obj.get_total_grid(), "total_grid");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_n_reduced_basis());
            obj.to_reduced_basis(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Full-grid field(s) -> reduced basis. A 1-D input gives a 1-D output, an (n_fields, total_grid) input an (n_fields, n_reduced) output.")
        .def("symmetrize_reduced_basis", [](SpaceGroup& obj, py::array_t<double, 0> reduced_field)
        {
            // Modified in place (like ComputationBox.zero_mean): a converted copy would
            // leave the caller's array unchanged, so it must already be C-contiguous float64.
            if (!(reduced_field.flags() & py::array::c_style))
                throw_with_line_number("symmetrize_reduced_basis modifies the array in place; pass a C-contiguous float64 array.");
            py::buffer_info buf = reduced_field.request();
            const int n_fields = sg_field_count(buf, obj.get_n_reduced_basis(), "n_reduced_basis");
            obj.symmetrize_reduced_basis(static_cast<double*>(buf.ptr), n_fields);
            return reduced_field;
        }, py::arg("reduced_field").noconvert(),
           "Symmetrize reduced-basis field(s) in place over the full space-group orbits; returns the same array.")
        .def("reduced_to_irreducible", [](SpaceGroup& obj, array_in<double> reduced_field)
        {
            py::buffer_info buf = reduced_field.request();
            const int n_fields = sg_field_count(buf, obj.get_n_reduced_basis(), "n_reduced_basis");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_n_reduced_basis_full());
            obj.reduced_to_irreducible(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Reduced-basis field(s) -> irreducible basis (one value per full-group orbit). A 1-D input gives a 1-D output, an (n_fields, n_reduced) input an (n_fields, n_irreducible) output.")
        .def("irreducible_to_reduced", [](SpaceGroup& obj, array_in<double> irreducible_field)
        {
            py::buffer_info buf = irreducible_field.request();
            const int n_fields = sg_field_count(buf, obj.get_n_reduced_basis_full(), "n_irreducible_basis");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_n_reduced_basis());
            obj.irreducible_to_reduced(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Irreducible-basis field(s) -> reduced (physical) basis. A 1-D input gives a 1-D output, an (n_fields, n_irreducible) input an (n_fields, n_reduced) output.")
        .def("from_reduced_basis", [](SpaceGroup& obj, array_in<double> reduced_field)
        {
            py::buffer_info buf = reduced_field.request();
            const int n_fields = sg_field_count(buf, obj.get_n_reduced_basis(), "n_reduced_basis");
            py::array_t<double> out = sg_output_like(buf, n_fields, obj.get_total_grid());
            obj.from_reduced_basis(static_cast<double*>(buf.ptr), static_cast<double*>(out.request().ptr), n_fields);
            return out;
        }, "Reduced-basis field(s) -> full grid. A 1-D input gives a 1-D output, an (n_fields, n_reduced) input an (n_fields, total_grid) output.")
        .def("symmetrize", [](SpaceGroup& obj, array_in<double> field)
        {
            py::buffer_info buf = field.request();
            const int n_fields = sg_field_count(buf, obj.get_total_grid(), "total_grid");
            py::array_t<double> result(buf.shape);   // same shape as the input
            obj.symmetrize(static_cast<double*>(buf.ptr), static_cast<double*>(result.request().ptr), n_fields);
            return result;
        }, "Average field(s) over the space-group orbits. Returns an array of the same shape as the input.");

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
        .def_static("avail_platforms", &PlatformSelector::avail_platforms)
        .def_static("create_factory", [](std::string platform_name, bool reduce_memory,
                                         std::string type)
        {
            // Converting type to lowercase
            std::transform(type.begin(), type.end(), type.begin(),
            [](unsigned char c){ return std::tolower(c); });

            if (type == "real")
            {
                return py::cast(PlatformSelector::create_factory_real(platform_name, reduce_memory));
            }
            else if (type == "complex")
            {
                return py::cast(PlatformSelector::create_factory_complex(platform_name, reduce_memory));
            }
            else {
                throw std::runtime_error("Invalid type parameter. Must be either 'real' or 'complex'");
            }
        }, py::arg("platform_name"), py::arg("reduce_memory") = false,
           py::arg("type") = "real");
}
