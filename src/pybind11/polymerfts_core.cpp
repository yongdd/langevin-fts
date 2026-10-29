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
 * - FourierTransform (Real/Complex): Fourier multipliers of whole fields
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
 * An input array is read in place when it is C-contiguous and of the expected
 * dtype; other arrays (strided views, float32, ...) are first copied into a
 * converted temporary (see array_in). Functions that modify an array in place
 * (ComputationBox.zero_mean, SpaceGroup.symmetrize_reduced_basis) reject arrays
 * that would need such a copy. Results are returned as new NumPy arrays.
 * Array sizes are checked against the grid.
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
#include <numbers>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "BuildInfo.h"
#include "NumericalMethods.h"
#ifdef USE_CPU_FFTW
#include <fftw3.h>
#endif
#ifdef USE_CPU_MKL
#include <mkl.h>
#endif
#ifdef USE_CUDA
#include "CudaCommon.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#endif

#include "GridLimits.h"
#include "MinimalFormAlias.h"
#include "ContourLengthMapping.h"
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
 * Exact periodic metric for Python kernels, without creating a platform or FFT.
 * The optional moments <(Gm)_a (Gm)_b> average all tied minimal aliases, so
 * derivatives of the metric use the same representatives as the propagators.
 */
py::tuple periodic_metric(const std::vector<int>& nx, array_in<double> reciprocal_metric,
                          bool with_moments)
{
    const int dim = static_cast<int>(nx.size());
    const auto metric = reciprocal_metric.request();
    if (dim < 1 || dim > 3 || metric.ndim != 2 || metric.shape[0] != dim || metric.shape[1] != dim)
        throw py::value_error("A periodic metric needs 1 to 3 grid sizes and a matching square reciprocal metric.");
    long long n_grid = 1;
    std::array<int, 3> grid = {1, 1, 1};
    for (int d = 0; d < dim; ++d)
    {
        if (nx[d] < 1 || nx[d] > MAX_GRID_POINTS / n_grid)
            throw py::value_error("The periodic metric grid is empty or exceeds the supported grid size.");
        n_grid *= nx[d];
        grid[d] = nx[d];
    }
    const auto* values = static_cast<const double*>(metric.ptr);
    double G[3][3] = {};
    double scale = 0.0;
    for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
        {
            G[a][b] = values[a * dim + b];
            if (!std::isfinite(G[a][b]))
                throw py::value_error("The reciprocal metric must be finite.");
            scale = std::max(scale, std::abs(G[a][b]));
        }
    if (!(scale > 0.0))
        throw py::value_error("The reciprocal metric must be positive definite.");
    double normalized[3][3] = {};
    for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
        {
            if (std::abs(G[a][b] - G[b][a]) > 1e-12 * scale)
                throw py::value_error("The reciprocal metric must be symmetric.");
            normalized[a][b] = G[a][b] / scale;
        }
    const double minor2 = normalized[0][0] * normalized[1][1] - normalized[0][1] * normalized[1][0];
    const double determinant = normalized[0][0] * (normalized[1][1] * normalized[2][2] - normalized[1][2] * normalized[2][1])
                             - normalized[0][1] * (normalized[1][0] * normalized[2][2] - normalized[1][2] * normalized[2][0])
                             + normalized[0][2] * (normalized[1][0] * normalized[2][1] - normalized[1][1] * normalized[2][0]);
    if (!(normalized[0][0] > 0.0) || (dim >= 2 && !(minor2 > 0.0)) || (dim == 3 && !(determinant > 0.0)))
        throw py::value_error("The reciprocal metric must be positive definite.");

    py::array_t<double> k_squared(nx);
    py::array_t<double> moments;
    double* moment_data = nullptr;
    if (with_moments)
    {
        std::vector<py::ssize_t> shape = {dim, dim};
        shape.insert(shape.end(), nx.begin(), nx.end());
        moments = py::array_t<double>(shape);
        moment_data = moments.mutable_data();
    }
    auto* k_data = k_squared.mutable_data();
    const IndexMetric index_metric{G[0][0], G[1][1], G[2][2], G[0][1], G[0][2], G[1][2]};
    const double four_pi_squared = 4.0 * std::numbers::pi * std::numbers::pi;
    auto form = [&](int a, int b, int c) {
        return G[0][0] * a * a + G[1][1] * b * b + G[2][2] * c * c
             + 2.0 * (G[0][1] * a * b + G[0][2] * a * c + G[1][2] * b * c);
    };
    auto signed_index = [](int i, int n) { return i > n / 2 ? i - n : i; };
    {
        py::gil_scoped_release release;
        for (int i = 0; i < grid[0]; ++i)
            for (int j = 0; j < grid[1]; ++j)
                for (int k = 0; k < grid[2]; ++k)
                {
                    const long long index = (static_cast<long long>(i) * grid[1] + j) * grid[2] + k;
                    const std::array<int, 3> m = {signed_index(i, grid[0]), signed_index(j, grid[1]), signed_index(k, grid[2])};
                    auto representative = m;
                    min_form_alias_rep(representative, grid, index_metric, form);
                    k_data[index] = four_pi_squared * form(representative[0], representative[1], representative[2]);
                    if (with_moments)
                    {
                        double products[3][3] = {};
                        const int count = for_each_min_form_alias(m, grid, index_metric, form, 1e-10,
                            [&](int a, int b, int c) {
                                double u[3] = {};
                                for (int d = 0; d < dim; ++d)
                                    u[d] = G[d][0] * a + G[d][1] * b + G[d][2] * c;
                                for (int row = 0; row < dim; ++row)
                                    for (int col = 0; col < dim; ++col)
                                        products[row][col] += u[row] * u[col];
                            });
                        for (int row = 0; row < dim; ++row)
                            for (int col = 0; col < dim; ++col)
                                moment_data[(row * dim + col) * n_grid + index] = products[row][col] / count;
                    }
                }
    }
    if (with_moments)
        return py::make_tuple(k_squared, moments);
    return py::make_tuple(k_squared, py::none());
}


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
 * - get_angles(), get_angles_degrees(), get_lattice_vectors(), is_orthogonal(), get_recip_metric(): Cell geometry
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
        .def("get_dv", [](const ComputationBox<T>& obj, int i)
        {
            if (i < 0 || i >= obj.get_total_grid())
                throw_with_line_number("Index (" + std::to_string(i) + ") must be in range [0, " + std::to_string(obj.get_total_grid() - 1) + "]");
            return obj.get_dv(i);
        })
        .def("get_total_grid", &ComputationBox<T>::get_total_grid)
        .def("get_volume", &ComputationBox<T>::get_volume)
        .def("get_accessible_volume", &ComputationBox<T>::get_accessible_volume)
        .def("get_angles", &ComputationBox<T>::get_angles)
        .def("get_angles_degrees", &ComputationBox<T>::get_angles_degrees)
        .def("get_lattice_vectors", [](const ComputationBox<T>& obj)
        {
            // Rows a, b, c in Cartesian coordinates (a along x, b in the xy plane)
            const std::array<double, 9>& v = obj.get_lattice_vec();
            return std::vector<std::vector<double>>{{v[0], v[1], v[2]}, {v[3], v[4], v[5]}, {v[6], v[7], v[8]}};
        })
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
            // (a wrong dtype is rejected by pybind11 before the call), and writable
            // (a read-only memory map would crash, an immutable buffer would change).
            if (!(g.flags() & py::array::c_style) || !g.writeable())
                throw_with_line_number("zero_mean modifies the array in place; pass a writable C-contiguous array.");
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
 * - compute_partition_errors(): Relative partition function error of each polymer
 * - check_total_partition(tolerance): Whether every error is within the tolerance
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

                std::map<std::string, const T*> map_buf_w_input;
                std::map<std::string, const T*> map_buf_q_init;
                // Casts can own temporary dtype/contiguity conversions. Keep them
                // alive until the computation has finished reading their buffers.
                std::map<std::string, array_in<const T>> q_init_map;

                // Validate and collect w_input
                for (auto it = w_input.begin(); it != w_input.end(); ++it)
                {
                    py::buffer_info buf_w_input = it->second.request();
                    if (buf_w_input.size != N) {
                        throw_with_line_number("Size of input w[" + it->first + "] (" + std::to_string(buf_w_input.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
                    }
                    map_buf_w_input[it->first] = (const T*)buf_w_input.ptr;
                }

                // Handle q_init if provided: in the same basis as w (reduced with a space group),
                // which is the layout the computations copy
                if (!q_init.is_none()) {
                    q_init_map = q_init.cast<std::map<std::string, array_in<const T>>>();

                    for (auto it = q_init_map.begin(); it != q_init_map.end(); ++it)
                    {
                        py::buffer_info buf_q_init = it->second.request();
                        if (buf_q_init.size != N) {
                            throw_with_line_number("Size of input q[" + it->first + "] (" + std::to_string(buf_q_init.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
                        }
                        map_buf_q_init[it->first] = (const T*)buf_q_init.ptr;
                    }
                }

                obj.compute_propagators(map_buf_w_input, map_buf_q_init);
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

                // Allocate output buffer: reduced basis with a space group, as in compute_propagators
                const int N = obj.get_cb()->get_n_basis();
                py::array_t<T> q_out(N);
                py::buffer_info buf_q_out = q_out.request();
                T* q_out_ptr = static_cast<T*>(buf_q_out.ptr);

                // Call C++ implementation
                if (buf_q_in.size != N)
                {
                    throw_with_line_number("Size of input q_in (" + std::to_string(buf_q_in.size) + ") and 'n_grid' (" + std::to_string(N) + ") must match.");
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
        .def("compute_partition_errors", &PropagatorComputation<T>::compute_partition_errors,
            "Relative partition function error of each polymer (nothing is printed)")
        .def("check_total_partition", &PropagatorComputation<T>::check_total_partition,
            py::arg("tolerance") = PARTITION_TOLERANCE,
            "Whether every error of compute_partition_errors() is finite and at most tolerance");
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
 * @brief Bind FourierTransform<T> and its FourierMultiplier<T> handles to Python.
 *
 * Creates the Python classes "FourierTransform_{Real|Complex}" and
 * "FourierMultiplier_{Real|Complex}":
 * - coefficient_shape, complex_coefficients: Layout of the coefficients of a field
 * - field_type: "real" or "complex", the fields the transform takes; periodic: FFT (or wall series)
 * - k_squared(): |k|² in the coefficient shape, from the current box
 * - set_multiplier(values): Store a multiplier of shape coefficient_shape
 *   (diagonal: applied to each field) or (n_out, n_in) + coefficient_shape
 * - apply([(multiplier, fields), ...]): backward(Σ M · forward(fields))
 * - forward(fields), backward(coefficients): Transforms of one or more fields
 *
 * Fields are flat (n_grid,) or (n_fields, n_grid); the output of apply() is flat
 * when every input is flat and there is one output field.
 *
 * @tparam T Field type (double or std::complex<double>)
 * @param m Python module
 * @param type_name Suffix ("Real" or "Complex")
 */
template<typename T>
void bind_fourier_transform(py::module &m, const std::string &type_name)
{
    // Number of fields of a flat (n_grid,) or (n_fields, n_grid) array
    auto field_count = [](const py::buffer_info& buf, int n_grid) -> int
    {
        if ((buf.ndim != 1 && buf.ndim != 2) || buf.shape[buf.ndim - 1] != n_grid || buf.size == 0)
            throw_with_line_number("Fields must be shaped (" + std::to_string(n_grid) + ",) or (n_fields, "
                + std::to_string(n_grid) + ").");
        return buf.ndim == 1 ? 1 : static_cast<int>(buf.shape[0]);
    };
    // Number of fields of an array of coefficients: coefficient_shape or (n_fields,) + coefficient_shape
    auto coefficient_count = [](const py::buffer_info& buf, const std::vector<int>& shape) -> int
    {
        const int lead = static_cast<int>(buf.ndim) - static_cast<int>(shape.size());
        bool ok = (lead == 0 || lead == 1) && buf.size > 0;
        for (size_t d = 0; ok && d < shape.size(); ++d)
            ok = buf.shape[lead + d] == shape[d];
        if (!ok)
            throw_with_line_number("Coefficients must have the coefficient shape, optionally after the number of fields.");
        return lead == 0 ? 1 : static_cast<int>(buf.shape[0]);
    };
    auto shaped = [](std::vector<py::ssize_t> lead, const std::vector<int>& shape)
    {
        for (int n : shape)
            lead.push_back(n);
        return lead;
    };
    // The conversion to a real array would drop the imaginary part (with only a ComplexWarning)
    auto reject_complex = [](py::handle values, bool real, const char* what)
    {
        if (real && py::module_::import("numpy").attr("iscomplexobj")(values).cast<bool>())
            throw_with_line_number(std::string(what) + " of this transform are real: got a complex array. "
                "Complex fields need a solver created with field_type=\"complex\".");
    };
    constexpr bool real_fields = std::is_same_v<T, double>;

    py::class_<FourierMultiplier<T>>(m, ("FourierMultiplier_" + type_name).c_str())
        .def_property_readonly("n_out", &FourierMultiplier<T>::get_n_out)
        .def_property_readonly("n_in", &FourierMultiplier<T>::get_n_in)
        .def_property_readonly("diagonal", &FourierMultiplier<T>::is_diagonal);

    py::class_<FourierTransform<T>>(m, ("FourierTransform_" + type_name).c_str())
        .def_property_readonly("coefficient_shape", [](FourierTransform<T>& obj)
        {
            return py::tuple(py::cast(obj.get_coefficient_shape()));
        })
        .def_property_readonly("complex_coefficients", &FourierTransform<T>::has_complex_coefficients)
        .def_property_readonly("field_type", [](FourierTransform<T>&) { return real_fields ? "real" : "complex"; })
        .def_property_readonly("periodic", [](FourierTransform<T>& obj)
        {
            return obj.get_boundary_conditions()[0] == BoundaryCondition::PERIODIC;   // all axes alike
        })
        .def("k_squared", [](FourierTransform<T>& obj)
        {
            const std::vector<double> k_sq = obj.get_k_squared();
            py::array_t<double> result(obj.get_coefficient_shape());
            std::copy(k_sq.begin(), k_sq.end(), result.mutable_data());
            return result;
        })
        // The handle keeps its transform alive (it is used only with that transform)
        .def("set_multiplier", [](FourierTransform<T>& obj, array_in<std::complex<double>> values)
        {
            py::buffer_info buf = values.request();
            const std::vector<int>& shape = obj.get_coefficient_shape();
            const int lead = static_cast<int>(buf.ndim) - static_cast<int>(shape.size());
            bool ok = lead == 0 || lead == 2;
            for (size_t d = 0; ok && d < shape.size(); ++d)
                ok = buf.shape[lead + d] == shape[d];
            if (!ok || buf.size == 0)
                throw_with_line_number("A multiplier is shaped like the coefficients (diagonal) "
                    "or (n_out, n_in) + coefficient_shape.");
            const bool diagonal = lead == 0;
            const int n_out = diagonal ? 1 : static_cast<int>(buf.shape[0]);
            const int n_in = diagonal ? 1 : static_cast<int>(buf.shape[1]);
            return obj.create_multiplier(static_cast<const std::complex<double>*>(buf.ptr), n_out, n_in, diagonal);
        }, py::arg("values"), py::keep_alive<0, 1>())
        .def("apply", [field_count, reject_complex](FourierTransform<T>& obj, py::list terms)
        {
            const int n_grid = obj.get_n_grid();
            std::vector<array_in<T>> inputs;   // keeps converted inputs alive
            std::vector<FourierTerm<T>> fourier_terms;
            bool all_flat = true;
            for (py::handle item : terms)
            {
                py::tuple pair = py::cast<py::tuple>(item);
                if (pair.size() != 2)
                    throw_with_line_number("Each term is a (multiplier, fields) pair.");
                const FourierMultiplier<T>* multiplier = pair[0].cast<const FourierMultiplier<T>*>();
                if (multiplier == nullptr)
                    throw_with_line_number("Each term needs a multiplier made by set_multiplier().");
                reject_complex(pair[1], real_fields, "The fields");
                inputs.push_back(pair[1].cast<array_in<T>>());
                py::buffer_info buf = inputs.back().request();
                const int n_fields = field_count(buf, n_grid);
                all_flat = all_flat && buf.ndim == 1;
                fourier_terms.push_back({multiplier, static_cast<const T*>(buf.ptr), n_fields});
            }
            if (fourier_terms.empty())
                throw_with_line_number("apply() needs at least one term.");
            const FourierTerm<T>& first = fourier_terms[0];
            const int n_out = first.multiplier->is_diagonal() ? first.n_fields : first.multiplier->get_n_out();
            py::array_t<T> result = (all_flat && n_out == 1)
                ? py::array_t<T>(n_grid) : py::array_t<T>({static_cast<py::ssize_t>(n_out), static_cast<py::ssize_t>(n_grid)});
            obj.apply(fourier_terms, n_out, result.mutable_data());
            return result;
        }, py::arg("terms"))
        .def("forward", [field_count, shaped, reject_complex](FourierTransform<T>& obj, py::object values) -> py::array
        {
            reject_complex(values, real_fields, "The fields");
            array_in<T> fields = values.cast<array_in<T>>();
            py::buffer_info buf = fields.request();
            const int n_fields = field_count(buf, obj.get_n_grid());
            const std::vector<py::ssize_t> lead = buf.ndim == 1 ? std::vector<py::ssize_t>{} : std::vector<py::ssize_t>{n_fields};
            const std::vector<py::ssize_t> shape = shaped(lead, obj.get_coefficient_shape());
            if (obj.has_complex_coefficients())
            {
                py::array_t<std::complex<double>> result(shape);
                obj.forward(static_cast<const T*>(buf.ptr), n_fields, reinterpret_cast<double*>(result.mutable_data()));
                return result;
            }
            py::array_t<double> result(shape);
            obj.forward(static_cast<const T*>(buf.ptr), n_fields, result.mutable_data());
            return result;
        }, py::arg("fields"))
        .def("backward", [coefficient_count, reject_complex](FourierTransform<T>& obj, py::object coefficients)
        {
            const int n_grid = obj.get_n_grid();
            auto run = [&](auto input, const double* values)
            {
                py::buffer_info buf = input.request();
                const int n_fields = coefficient_count(buf, obj.get_coefficient_shape());
                const bool flat = buf.ndim == static_cast<py::ssize_t>(obj.get_coefficient_shape().size());
                py::array_t<T> result = flat ? py::array_t<T>(n_grid)
                    : py::array_t<T>({static_cast<py::ssize_t>(n_fields), static_cast<py::ssize_t>(n_grid)});
                obj.backward(values, n_fields, result.mutable_data());
                return result;
            };
            if (obj.has_complex_coefficients())
            {
                array_in<std::complex<double>> input = coefficients.cast<array_in<std::complex<double>>>();
                return run(input, reinterpret_cast<const double*>(input.data()));
            }
            reject_complex(coefficients, true, "The coefficients");
            array_in<double> input = coefficients.cast<array_in<double>>();
            return run(input, input.data());
        }, py::arg("coefficients"));
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
 * - create_fourier_transform(cb, bc): Create a Fourier transform (one transform per axis)
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
            py::keep_alive<0, 2>(), py::keep_alive<0, 3>(), py::keep_alive<0, 4>(), py::keep_alive<0, 6>(),
            py::keep_alive<2, 6>())  // the constructor also sets the space group on the box
        .def("create_anderson_mixing", [](AbstractFactory<T>& obj, long long n_var, int max_hist,
                                          double start_error, double mix_min, double mix_init)
        {
            // A Python size can exceed int: check it before narrowing
            check_anderson_variables(n_var);
            return obj.create_anderson_mixing(static_cast<int>(n_var), max_hist, start_error, mix_min, mix_init);
        }, py::arg("n_var"), py::arg("max_hist"), py::arg("start_error"), py::arg("mix_min"), py::arg("mix_init"))
        .def("create_fourier_transform", [](AbstractFactory<T>& obj, ComputationBox<T>* cb, std::vector<std::string> bc)
        {
            // One transform per axis: "periodic", "reflecting" (cosine) or "absorbing" (sine)
            std::vector<BoundaryCondition> bc_vec;
            for (std::string name : bc)
            {
                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
                if (name == "periodic")
                    bc_vec.push_back(BoundaryCondition::PERIODIC);
                else if (name == "reflecting")
                    bc_vec.push_back(BoundaryCondition::REFLECTING);
                else if (name == "absorbing")
                    bc_vec.push_back(BoundaryCondition::ABSORBING);
                else
                    throw_with_line_number("Unknown transform '" + name + "': use periodic, reflecting or absorbing.");
            }
            return obj.create_fourier_transform(cb, bc_vec);
        }, py::arg("cb"), py::arg("bc"), py::keep_alive<0, 2>())
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
 * - FourierTransform_Real, FourierTransform_Complex (with FourierMultiplier_*)
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
    if (in.ndim != 1 && in.ndim != 2)
        throw_with_line_number("Input must be flat (1-D) or shaped (n_fields, n_grid).");
    if (in.ndim == 2 && in.shape[1] != n_in)
        throw_with_line_number("The last input axis must equal " + std::string(n_in_name) +
            " (" + std::to_string(n_in) + ").");
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

// Converts the Python block list [[monomer_type, contour_length, v, u], ...]
static std::vector<BlockInput> to_block_inputs(const std::vector<std::vector<py::object>>& block_list)
{
    if (block_list.empty())
        throw_with_line_number("A polymer needs at least one block.");
    std::vector<BlockInput> block_inputs;
    for (size_t b = 0; b < block_list.size(); b++)
    {
        const auto& py_block = block_list[b];
        if (py_block.size() != 4)
            throw_with_line_number("Block " + std::to_string(b) + " has " + std::to_string(py_block.size()) +
                " entries; a block is [monomer_type, contour_length, v, u].");
        BlockInput block;
        block.monomer_type = py::cast<std::string>(py_block[0]);
        block.contour_length = py::cast<double>(py_block[1]);
        block.v = py::cast<int>(py_block[2]);
        block.u = py::cast<int>(py_block[3]);
        block_inputs.push_back(block);
    }
    return block_inputs;
}

PYBIND11_MODULE(_core, m)
{
    m.def("_discrete_contour_length", &ContourLengthMapping::discrete_contour_length,
          py::arg("contour_length"), py::arg("ds"),
          "Validate a discrete block length and return the native canonical N*ds length.");
    m.def("_periodic_metric", &periodic_metric, py::arg("nx"), py::arg("reciprocal_metric"),
          py::arg("with_moments") = false,
          "Exact full-grid periodic k squared and optional tied-alias metric moments.");
    m.attr("build_info") = polymerfts_build_info();
    m.def("numerical_methods", []() {
        py::dict result;
        for (const auto& info : numerical_methods)
        {
            py::dict entry;
            entry["chain_model"] = info.chain_model;
            entry["solver_type"] = info.solver_type;
            entry["contour_order"] = info.contour_order;
            entry["supports_complex"] = info.supports_complex;
            entry["requires_orthogonal_cell"] = info.requires_orthogonal_cell;
            entry["supports_mixed_boundaries"] = info.supports_mixed_boundaries;
            py::list backends;
#ifdef USE_CPU_MKL
            backends.append("cpu-mkl");
#endif
#ifdef USE_CPU_FFTW
            backends.append("cpu-fftw");
#endif
#ifdef USE_CUDA
            backends.append("cuda");
#endif
            entry["compiled_backends"] = backends;
            result[py::str(info.name)] = entry;
        }
        return result;
    }, "Registered propagator methods and their compiled capabilities.");
    m.def("resolve_numerical_method", [](const std::string& chain_model, const std::string& numerical_method) {
        return std::string(resolve_numerical_method(chain_model, numerical_method).name);
    }, py::arg("chain_model"), py::arg("numerical_method") = "",
       "The method the factories use: 'discrete' for discrete chains, rqm4 for an empty name.");
    m.def("runtime_info", [](bool include_cuda) {
        py::dict info;
#ifdef USE_CPU_FFTW
        info["fftw"] = std::string(fftw_version);
#endif
#ifdef USE_CPU_MKL
        char mkl_version[256] = {};
        MKL_Get_Version_String(mkl_version, sizeof(mkl_version));
        info["mkl"] = std::string(mkl_version);
#endif
#ifdef USE_CUDA
        if (include_cuda)
        {
            int version = 0;
            CUDA_CHECK(cudaRuntimeGetVersion(&version));
            info["cuda_runtime"] = version;
            CUDA_CHECK(cudaDriverGetVersion(&version));
            info["cuda_driver_api"] = version;
            CUFFT_CHECK(cufftGetVersion(&version));
            info["cufft"] = version;
            int device = 0;
            cudaDeviceProp properties{};
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
            info["cuda_device_name"] = std::string(properties.name);
            info["cuda_compute_capability"] = std::to_string(properties.major) + "." + std::to_string(properties.minor);
            info["cuda_device_memory"] = properties.totalGlobalMem;
            std::string uuid;
            constexpr char hex[] = "0123456789abcdef";
            for (unsigned char byte : properties.uuid.bytes)
            {
                uuid.push_back(hex[byte >> 4]);
                uuid.push_back(hex[byte & 15]);
            }
            info["cuda_device_uuid"] = uuid;
        }
#else
        (void)include_cuda;
#endif
        return info;
    }, py::arg("include_cuda") = false);
    // Grid-size limits of the 32-bit grid indices (GridLimits.h)
    m.attr("MAX_GRID_POINTS") = MAX_GRID_POINTS;
    m.attr("MAX_GRID_ARRAY") = MAX_GRID_ARRAY;
    // Default tolerance of check_total_partition (PropagatorComputation.h)
    m.attr("PARTITION_TOLERANCE") = PARTITION_TOLERANCE;


    py::class_<Block>(m, "Block")
        .def_readonly("monomer_type", &Block::monomer_type)
        .def_readonly("n_segment", &Block::n_segment)
        .def_readonly("contour_length", &Block::contour_length)
        .def_readonly("v", &Block::v)
        .def_readonly("u", &Block::u);

    py::class_<ComputationEdge>(m, "ComputationEdge")
        .def_readonly("max_n_segment", &ComputationEdge::max_n_segment)
        .def_readonly("monomer_type", &ComputationEdge::monomer_type)
        .def_readonly("deps", &ComputationEdge::deps)
        .def_readonly("height", &ComputationEdge::height)
        .def_readonly("junction_ends", &ComputationEdge::junction_ends);

    py::class_<ComputationBlock>(m, "ComputationBlock")
        .def_readonly("monomer_type", &ComputationBlock::monomer_type)
        .def_readonly("n_segment_left", &ComputationBlock::n_segment_left)
        .def_readonly("n_segment_right", &ComputationBlock::n_segment_right)
        .def_readonly("n_repeated", &ComputationBlock::n_repeated)
        .def_readonly("v_u", &ComputationBlock::v_u);

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
            obj.add_polymer(volume_fraction, to_block_inputs(block_list), {});
        })
        .def("add_polymer", [](Molecules& obj, double volume_fraction, std::vector<std::vector<py::object>> block_list, std::map<int, std::string> chain_end_to_q_init)
        {
            obj.add_polymer(volume_fraction, to_block_inputs(block_list), chain_end_to_q_init);
        })
        .def("get_polymer", &Molecules::get_polymer)
        .def("get_n_solvent_types", &Molecules::get_n_solvent_types)
        .def("get_solvent", &Molecules::get_solvent)
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
        .def("check_cell", &SpaceGroup::check_cell, py::arg("lx"), py::arg("angles"),
             "Raise unless the cell (lengths, angles in degrees) belongs to the crystal system of the group.")
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
        .def("display_info", &SpaceGroup::display_info,
             "Print the group, its crystal system and the sizes of its bases.")
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
            // leave the caller's array unchanged, so it must already be C-contiguous float64,
            // and writable.
            if (!(reduced_field.flags() & py::array::c_style) || !reduced_field.writeable())
                throw_with_line_number("symmetrize_reduced_basis modifies the array in place; pass a writable C-contiguous float64 array.");
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

    // Bind FourierTransform
    bind_fourier_transform<double>(m, "Real");
    bind_fourier_transform<std::complex<double>>(m, "Complex");

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

            // The unique_ptr hands the factory to the Python object
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
