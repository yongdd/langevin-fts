/** Public C++ mutations must preserve the computation's buffer and geometry contracts. */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "AbstractFactory.h"
#include "PlatformSelector.h"

template <typename Call>
void rejects(Call call)
{
    try
    {
        call();
    }
    catch (const std::exception&)
    {
        return;
    }
    throw std::runtime_error("An incompatible computation state was accepted.");
}

void check_space_group(const std::string& platform, bool reduced, bool with_group)
{
    std::unique_ptr<AbstractFactory<double>> factory(PlatformSelector::create_factory_real(platform, reduced));
    std::unique_ptr<Molecules> mol(factory->create_molecules_information("continuous", 0.05, {{"A", 1.0}}));
    mol->add_polymer(1.0, {{"A", 1.0, 0, 1}});
    std::unique_ptr<PropagatorComputationOptimizer> opt(factory->create_propagator_computation_optimizer(mol.get(), false));
    std::unique_ptr<ComputationBox<double>> cb(factory->create_computation_box({8, 8, 8}, {3.0, 3.0, 3.0}, {}, nullptr));
    SpaceGroup initial({8, 8, 8}, "Im-3m", 529), replacement({8, 8, 8}, "P1", 1);
    SpaceGroup* group = with_group ? &initial : nullptr;
    std::unique_ptr<PropagatorComputation<double>> pc(factory->create_propagator_computation(
        cb.get(), mol.get(), opt.get(), "rqm4", group));
    const int n = cb->get_n_basis();
    std::vector<double> full(512), w(n), before(n), after(n);
    for (int i = 0; i < 512; ++i)
        full[i] = std::sqrt(2.0)*5.0*std::sin(i);
    if (group != nullptr)
    {
        group->symmetrize(full.data(), 1);
        group->to_reduced_basis(full.data(), w.data(), 1);
    }
    else
        w = full;
    pc->compute_propagators({{"A", w.data()}});
    pc->compute_concentrations();
    pc->get_total_concentration("A", before.data());

    rejects([&] { pc->set_space_group(&replacement); });
    if (with_group)
        rejects([&] { pc->set_space_group(nullptr); });
    else
        rejects([&] { pc->set_space_group(&initial); });
    if (pc->get_space_group() != group || cb->get_space_group() != group || cb->get_n_basis() != n)
        throw std::runtime_error("Rejected space-group change modified the computation or box.");

    // The same group can be reattached after another user of the box changed it.
    cb->set_space_group(&replacement);
    rejects([&] { pc->get_total_partition(0); });
    pc->set_space_group(group);
    pc->get_total_concentration("A", after.data());
    if (before != after || !pc->check_total_partition() || !(std::abs(cb->mean(after.data()) - 1.0) <= 1e-10))
        throw std::runtime_error("The original computation was not preserved after rejecting a group change.");
}

void check_cnadi(const std::string& platform, bool reduced)
{
    std::unique_ptr<AbstractFactory<double>> factory(PlatformSelector::create_factory_real(platform, reduced));
    std::unique_ptr<Molecules> mol(factory->create_molecules_information("continuous", 0.05, {{"A", 1.0}}));
    mol->add_polymer(1.0, {{"A", 1.0, 0, 1}});
    std::unique_ptr<PropagatorComputationOptimizer> opt(factory->create_propagator_computation_optimizer(mol.get(), false));
    std::unique_ptr<ComputationBox<double>> cb(factory->create_computation_box({8, 8}, {4.0, 4.0}, {}, nullptr));
    std::unique_ptr<PropagatorComputation<double>> pc(factory->create_propagator_computation(
        cb.get(), mol.get(), opt.get(), "cn-adi2"));
    std::vector<double> w(64), phi(64);
    for (int i = 0; i < 64; ++i)
        w[i] = std::sqrt(2.0)*5.0*std::sin(i);
    pc->compute_propagators({{"A", w.data()}});
    const double partition = pc->get_total_partition(0);

    // C++ callers can mutate the shared box directly. While this computation
    // exists, the box rejects unsupported angles before changing anything.
    rejects([&] { cb->set_lattice_parameters({4.0, 4.0}, {90.0, 90.0, 60.0}); });
    if (!cb->is_orthogonal() || cb->get_lx(0) != 4.0 || cb->get_lx(1) != 4.0)
        throw std::runtime_error("A rejected lattice change modified the box.");
    cb->set_lattice_parameters({4.0, 4.0}, {90.0, 90.0, 90.0});
    pc->update_laplacian_operator();
    pc->compute_propagators({{"A", w.data()}});
    pc->compute_concentrations();
    pc->get_total_concentration("A", phi.data());
    if (!(std::abs(pc->get_total_partition(0)/partition - 1.0) <= 1e-12) ||
        !(std::abs(cb->mean(phi.data()) - 1.0) <= 1e-10) || !pc->check_total_partition())
        throw std::runtime_error("CN-ADI did not recover after restoring the orthogonal box.");

    // The box is released with the computation
    pc.reset();
    cb->set_lattice_parameters({4.0, 4.0}, {90.0, 90.0, 60.0});
    if (cb->is_orthogonal())
        throw std::runtime_error("The box stayed orthogonal after the CN-ADI computation was destroyed.");
}

int main()
{
    try
    {
        std::vector<std::string> platforms;
#ifdef USE_CPU_FFTW
        platforms.push_back("cpu-fftw");
#endif
#ifdef USE_CPU_MKL
        platforms.push_back("cpu-mkl");
#endif
#ifdef USE_CUDA
        platforms.push_back("cuda");
#endif
        for (const auto& platform : platforms)
            for (bool reduced : {false, true})
            {
                check_space_group(platform, reduced, false);
                check_space_group(platform, reduced, true);
                check_cnadi(platform, reduced);
            }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return -1;
    }
}
