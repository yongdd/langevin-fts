/**
 * @file SimpsonRule.h
 * @brief Simpson's rule quadrature coefficients for contour integration.
 *
 * This header provides the SimpsonRule class which generates quadrature
 * weights for numerical integration along the polymer contour. These weights
 * are used when computing monomer concentrations by integrating propagator
 * products over the chain length.
 *
 * **Quadrature Rules:**
 *
 * The class automatically selects the appropriate quadrature rule based on
 * the number of intervals N:
 *
 * - N = 0: Single point (weight = 1)
 * - N = 1: Trapezoidal rule (1/2, 1/2)
 * - N = 3: Simpson's 3/8 rule
 * - N even: Simpson's 1/3 rule (1/3, 4/3, 2/3, 4/3, ..., 1/3)
 * - N odd: Composite Simpson's 1/3 + 3/8 for last segment
 *
 * **Usage in Concentration Calculation:**
 *
 * The concentration integral is:
 *   φ(r) = (φ/Q) ∫₀ᴺ q(r,s) q†(r,N-s) ds
 *
 * With Simpson's rule:
 *   φ(r) ≈ (φ/Q) * ds * Σ_i w_i * q(r,i*ds) * q†(r,N-i*ds)
 *
 * where w_i are the Simpson coefficients returned by get_coeff().
 *
 * @see PropagatorComputation::compute_concentrations for usage
 *
 * @example
 * @code
 * // Get Simpson coefficients for N=100 contour steps
 * int N = 100;
 * std::vector<double> weights = SimpsonRule::get_coeff(N);
 *
 * // weights.size() == N + 1
 * // weights[0] = 1/3, weights[1] = 4/3, weights[2] = 2/3, ...
 *
 * // Use for integration
 * double integral = 0.0;
 * for (int i = 0; i <= N; i++) {
 *     integral += weights[i] * f[i] * ds;
 * }
 * @endcode
 */

#ifndef SIMPSON_RULE_H_
#define SIMPSON_RULE_H_

#include <cassert>

/**
 * @class SimpsonRule
 * @brief Static utility class for Simpson's rule quadrature coefficients.
 *
 * Provides quadrature weights for numerical integration with 4th-order
 * accuracy (for smooth integrands). The coefficients are suitable for
 * contour integrals along the polymer chain.
 *
 * @note All methods are static; do not instantiate this class.
 */
class SimpsonRule
{
public:
    /**
     * @brief Get Simpson's rule coefficients for N intervals.
     *
     * Returns a vector of N+1 weights for integrating over N intervals.
     * The integral is approximated as: sum_i w[i] * f(x_i) * dx
     *
     * @param N Number of intervals (must be >= 0)
     * @return Vector of N+1 quadrature weights
     *
     * @note Weights are normalized assuming uniform spacing.
     *       Multiply by interval width (ds) for actual integration.
     *
     * **Weight Patterns:**
     *
     * - N=0: [1.0]
     * - N=1: [0.5, 0.5] (trapezoidal)
     * - N=2: [1/3, 4/3, 1/3] (Simpson's 1/3)
     * - N=3: [3/8, 9/8, 9/8, 3/8] (Simpson's 3/8)
     * - N=4: [1/3, 4/3, 2/3, 4/3, 1/3]
     * - N=5: [1/3, 4/3, 17/24, 9/8, 9/8, 3/8] (composite)
     *
     * @example
     * @code
     * // Integrate f(x) from 0 to 1 with 10 intervals
     * double dx = 1.0 / 10;
     * auto w = SimpsonRule::get_coeff(10);
     *
     * double integral = 0.0;
     * for (int i = 0; i <= 10; i++) {
     *     double x = i * dx;
     *     integral += w[i] * f(x) * dx;
     * }
     * @endcode
     */
    static std::vector<double> get_coeff(const int N)
    {
        assert(N >= 0 && "N must be a non-negative number");
        std::vector<double> coeff;
        if ( N == 0)
            coeff.push_back(1.0);
        // Trapezoidal rule for single interval
        else if ( N == 1)
        {
            coeff.push_back(1.0/2.0);
            coeff.push_back(1.0/2.0);
        }
        // Simpson's 3/8 rule for 3 intervals
        else if ( N == 3)
        {
            coeff.push_back(3.0/8.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(3.0/8.0);
        }
        // Simpson's 1/3 rule for even number of intervals
        else if ( N % 2 == 0)
        {
            coeff.push_back(1.0/3.0);
            for(int n=1; n<=N-1; n++)
            {
                if ( n % 2 == 1)
                    coeff.push_back(4.0/3.0);
                else
                    coeff.push_back(2.0/3.0);
            }
            coeff.push_back(1.0/3.0);
        }
        // Composite Simpson's 1/3 + 3/8 for odd number of intervals
        else
        {
            coeff.push_back(1.0/3.0);
            for(int n=1; n<=N-4; n++)
            {
                if ( n % 2 == 1)
                    coeff.push_back(4.0/3.0);
                else
                    coeff.push_back(2.0/3.0);
            }
            coeff.push_back(3.0/8.0 + 1.0/3.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(9.0/8.0);
            coeff.push_back(3.0/8.0);
        }
        return coeff;
    };
};
#endif