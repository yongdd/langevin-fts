/** The GCE diagnostic uses the exact box quadrature, including masks and orbits. */
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "PlatformSelector.h"
#include "TestCudaPart.h"
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
#include "CpuComputationBackend.h"
#endif
#ifdef USE_CUDA
#include "CudaComputationBackend.h"
#endif

template <typename T>
void check_integrand(const std::string& platform, bool reduced_basis)
{
    std::unique_ptr<AbstractFactory<T>> factory;
    if constexpr (std::is_same_v<T, double>)
        factory = PlatformSelector::create_factory_real(platform, false);
    else
        factory = PlatformSelector::create_factory_complex(platform, false);

    const std::vector<int> nx = {8, 8, 8};
    SpaceGroup group(nx, "Im-3m", 529);
    std::vector<double> mask(512);
    for (int i = 0; i < 512; ++i)
        mask[i] = (i % 7 < 3) ? 1.0 : 0.0;
    if (reduced_basis)
    {
        // Select complete symmetry orbits while keeping the mask exactly binary.
        // Averaging a generic binary mask would produce unsupported fractions.
        std::fill(mask.begin(), mask.end(), 0.0);
        mask[0] = mask[1] = mask[2] = 1.0;
        group.symmetrize(mask.data(), 1);
        for (double& value : mask)
            value = value > 0.0 ? 1.0 : 0.0;
    }
    auto cb = factory->create_computation_box(nx, {2.0, 2.0, 2.0}, {}, mask.data());
    if (reduced_basis)
        cb->set_space_group(&group);
    auto molecules = factory->create_molecules_information("continuous", 0.1, {{"A", 1.0}});
    molecules->add_polymer(0.0, {{"A", 1.0, 0, 1}});
    const auto& method = resolve_numerical_method("continuous", "rqm4");
    std::unique_ptr<ComputationBackend<T>> backend;
#if defined(USE_CPU_FFTW) || defined(USE_CPU_MKL)
    if (platform == "cpu-fftw" || platform == "cpu-mkl")
        backend = std::make_unique<CpuComputationBackend<T>>(
            cb.get(), molecules.get(), method,
            platform == "cpu-fftw" ? FFTBackend::FFTW : FFTBackend::MKL,
            reduced_basis ? &group : nullptr, 2);
#endif
#ifdef USE_CUDA
    if (platform == "cuda")
        backend = std::make_unique<CudaComputationBackend<T>>(
            cb.get(), molecules.get(), method, reduced_basis ? &group : nullptr, 2);
#endif
    if (!backend)
        throw std::runtime_error("Missing backend in partition-integrand test.");

    const int n = cb->get_n_basis();
    std::vector<T> a(n), b(n), divisor(n), absolute(n);
    for (int i = 0; i < n; ++i)
    {
        a[i] = 2.0 + std::sin(i);
        b[i] = 1.0 + 0.3*std::cos(i);
        divisor[i] = 0.7 + 0.1*(i % 5);
        if constexpr (!std::is_same_v<T, double>)
        {
            a[i] *= std::polar(1.0, 0.7*i);
            divisor[i] *= std::polar(1.0, 0.2*i);
        }
    }
    auto release = [&](BackendArray<T>* ptr) { backend->deallocate(ptr); };
    std::unique_ptr<BackendArray<T>, decltype(release)> da(backend->allocate(), release);
    std::unique_ptr<BackendArray<T>, decltype(release)> db(backend->allocate(), release);
    std::unique_ptr<BackendArray<T>, decltype(release)> dw(backend->allocate(), release);
    const int other_stream = backend->get_n_streams() > 1 ? 1 : 0;
    backend->copy_from_host(0, da.get(), a.data());
    backend->copy_from_host(other_stream, db.get(), b.data());
    backend->copy_from_host(other_stream, dw.get(), divisor.data());

    for (double length : {2.0, 3.0})
    {
        cb->set_lx({length, length, length});
        for (bool inverse_weight : {false, true})
        {
            for (int i = 0; i < n; ++i)
                absolute[i] = std::abs(inverse_weight ? a[i]*b[i]/divisor[i] : a[i]*b[i]);
            const double expected = std::real(cb->integral(absolute.data()));
            const double actual = backend->absolute_inner_product(
                da.get(), db.get(), inverse_weight ? dw.get() : nullptr);
            if (!std::isfinite(actual) || std::abs(actual/expected - 1.0) > 2e-14)
                throw std::runtime_error("Absolute integrand lost mask/orbit/cell quadrature weights.");
            if constexpr (std::is_same_v<T, double>)
            {
                const double partition = inverse_weight
                    ? backend->inner_product_inverse_weight(da.get(), db.get(), dw.get())
                    : backend->inner_product(da.get(), db.get());
                if (std::abs(actual/partition - 1.0) > 2e-14)
                    throw std::runtime_error("Positive integrand scale differs from its partition.");
            }
        }
    }
}

int main()
{
    try
    {
        for (const auto& platform : PlatformSelector::avail_platforms())
        {
            check_integrand<double>(platform, false);
            check_integrand<double>(platform, true);
            check_integrand<std::complex<double>>(platform, false);
        }
        return cuda_part_missing() ? 1 : 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
