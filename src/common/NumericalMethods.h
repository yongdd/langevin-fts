/** Method names and capabilities shared by Python, factories and backends.
 * Add a descriptor here and a construction case in each computation backend.
 * The numerical algorithm itself stays in its CPU/CUDA solver classes.
 */
#ifndef POLYMERFTS_NUMERICAL_METHODS_H
#define POLYMERFTS_NUMERICAL_METHODS_H

#include <array>
#include <string>
#include <string_view>
#include "Exception.h"

enum class NumericalMethodId { RQM4, RK2, CNADI2, Discrete };

struct NumericalMethodInfo
{
    NumericalMethodId id;
    const char* name;
    const char* chain_model;
    const char* solver_type;
    int contour_order;  // 0: discrete chain; not a continuum MDE integrator
    bool supports_complex;
    bool requires_orthogonal_cell;
    bool supports_mixed_boundaries;
};

inline constexpr std::array<NumericalMethodInfo, 4> numerical_methods = {{
    {NumericalMethodId::RQM4, "rqm4", "continuous", "pseudospectral", 4, true, false, false},
    {NumericalMethodId::RK2, "rk2", "continuous", "pseudospectral", 2, true, false, false},
    {NumericalMethodId::CNADI2, "cn-adi2", "continuous", "realspace", 2, false, true, true},
    {NumericalMethodId::Discrete, "discrete", "discrete", "pseudospectral", 0, true, false, false}
}};

inline const NumericalMethodInfo& resolve_numerical_method(
    std::string_view chain_model, std::string_view name, std::string_view solver_type = "")
{
    // Discrete chains always use their own bond-convolution solver.
    if (chain_model == "discrete")
        name = "discrete";
    else if (chain_model != "continuous")
    {
        throw_with_line_number("Unknown chain model: " + std::string(chain_model));
    }
    else if (name.empty())
        name = solver_type == "realspace" ? "cn-adi2" : "rqm4";

    std::string valid;
    for (const auto& info : numerical_methods)
    {
        if (name == info.name && chain_model == info.chain_model)
            return info;
        if (chain_model == info.chain_model)
            valid += (valid.empty() ? "" : ", ") + std::string(info.name);
    }
    throw_with_line_number("Unknown numerical method '" + std::string(name) + "' for " + std::string(chain_model) +
        " chains (valid: " + valid + ").");
}

// mixed_boundaries: some directions periodic and others reflecting or absorbing
inline void validate_numerical_method(const NumericalMethodInfo& info, bool complex_fields, bool orthogonal,
                                      bool mixed_boundaries)
{
    if (complex_fields && !info.supports_complex)
        throw_with_line_number(std::string(info.name) + " does not support complex fields.");
    if (!orthogonal && info.requires_orthogonal_cell)
        throw_with_line_number(std::string(info.name) + " requires an orthogonal computation box.");
    if (mixed_boundaries && !info.supports_mixed_boundaries)
        throw_with_line_number(std::string(info.name) + " cannot mix periodic and non-periodic boundary "
            "conditions: make every direction periodic or every direction non-periodic, or use cn-adi2 "
            "(continuous chains).");
}

#endif
