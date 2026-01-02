"""Field compressor algorithms for L-FTS imaginary field saddle point finding.

This module provides compressor classes for finding saddle points of imaginary
auxiliary fields in Langevin Field-Theoretic Simulations.
"""

import numpy as np

class LR:
    """Linear Response (LR) field compressor for L-FTS.

    Uses the local Hessian (second derivative matrix) of the Hamiltonian
    to find the saddle point of imaginary auxiliary fields via linear response
    theory. This is faster than iterative methods but requires computing and
    inverting the Hessian matrix.

    The update is: w_new = w_old - J^(-1) · δH/δw, where J is the Jacobian
    (Hessian) in Fourier space.

    Parameters
    ----------
    nx : list of int
        Grid dimensions [nx, ny, nz].
    lx : list of float
        Box dimensions in units of a_Ref * N_Ref^(1/2).
    jk : ndarray
        Jacobian (Hessian) matrix in Fourier space. Precomputed from
        finite difference of δH/δw. Shape matches rfft output of nx.

    Attributes
    ----------
    nx : list of int
        Grid dimensions.
    lx : list of float
        Box dimensions.
    jk : ndarray
        Jacobian in Fourier space.

    Notes
    -----
    **Linear Response Theory:**

    The saddle point satisfies δH/δw = 0. Linearizing near the current point:

    .. math::
        \\frac{\\delta H}{\\delta w}(w + \\Delta w) \\approx \\frac{\\delta H}{\\delta w}(w) + J \\cdot \\Delta w

    Setting this to zero and solving for Δw:

    .. math::
        \\Delta w = -J^{-1} \\cdot \\frac{\\delta H}{\\delta w}

    The Jacobian J is diagonal in Fourier space, making inversion trivial.

    **Advantages:**

    - Single-step convergence if Hamiltonian is quadratic
    - No iteration needed
    - Very fast for simple systems

    **Disadvantages:**

    - Requires precomputed Hessian (expensive setup)
    - May not converge for highly nonlinear systems
    - Only works for single imaginary field (I=1)

    **Fourier Space Implementation:**

    The Jacobian is computed in Fourier space where it's diagonal:

    .. math::
        J(\\mathbf{k}) = \\frac{\\partial^2 H}{\\partial w(\\mathbf{k})^2}

    This enables efficient inversion: Δw(k) = -δH/δw(k) / J(k).

    See Also
    --------
    LRAM : Combined Linear Response + Anderson Mixing.
    LFTS.find_saddle_point : Uses this compressor.

    Examples
    --------
    >>> # LR is typically created inside LFTS.__init__
    >>> # with precomputed Hessian jk
    >>> lr = LR(nx=[32, 32, 32], lx=[4.0, 4.0, 4.0], jk=jacobian_k)
    >>>
    >>> # In saddle point iteration
    >>> w_new = lr.calculate_new_fields(w_current, -h_deriv, old_err, err)
    """
    def __init__(self, nx, lx, jk):
        self.nx = nx
        self.lx = lx
        self.jk = jk.copy()

        # self.jk_array_shape = nx.copy()
        # self.jk_array_shape[-1] = self.jk_array_shape[-1]//2 + 1
        # self.jk = np.zeros(self.jk_array_shape)

    def reset_count(self,):
        """Reset compressor state (no-op for LR).

        LR is stateless, so this method does nothing. Included for
        compatibility with Anderson Mixing interface.
        """
        pass

    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        """Compute new fields using linear response method.

        Performs one linear response update to move toward the saddle point.

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (1, total_grid) or (total_grid,).
        negative_h_deriv : ndarray
            Negative functional derivative -δH/δw, same shape as w_current.
        old_error_level : float
            Previous error level (not used by LR, kept for interface compatibility).
        error_level : float
            Current error level (not used by LR, kept for interface compatibility).

        Returns
        -------
        w_new : ndarray
            Updated field values, same shape as w_current.

        Notes
        -----
        The update is performed in Fourier space:

        1. Transform -δH/δw to Fourier space
        2. Divide by Jacobian: Δw(k) = -δH/δw(k) / J(k)
        3. Transform back to real space
        4. Update: w_new = w_current + Δw

        This assumes the Jacobian J(k) has already been precomputed and
        is stored in self.jk.
        """
        nx = self.nx
        w_diff = np.zeros_like(w_current)
        negative_h_deriv_k = np.fft.rfftn(np.reshape(negative_h_deriv, nx))/np.prod(nx)
        w_diff_k = negative_h_deriv_k/self.jk
        w_diff[0] = np.reshape(np.fft.irfftn(w_diff_k, nx), np.prod(nx))*np.prod(nx)

        return w_current + w_diff

class LRAM:
    """Linear Response + Anderson Mixing (LRAM) field compressor.

    Combines Linear Response for coarse optimization with Anderson Mixing for
    final refinement. This hybrid approach is more robust than LR alone while
    being faster than pure Anderson Mixing.

    The algorithm:
    1. Use LR to get approximate update
    2. Use AM to refine the LR result

    This combines the speed of LR with the robustness of AM.

    Parameters
    ----------
    lr : LR
        Linear Response compressor instance.
    am : AndersonMixing
        Anderson Mixing instance (from C++ backend).

    Attributes
    ----------
    lr : LR
        Linear Response compressor.
    am : AndersonMixing
        Anderson Mixing optimizer.

    Notes
    -----
    **Algorithm:**

    At each compression step:

    1. **LR step**: w_lr = w_old - J^(-1) · δH/δw
    2. **AM step**: w_new = AM(w_lr, w_lr - w_old)

    The AM step treats the LR update as a "gradient" and applies Anderson
    mixing history to accelerate convergence.

    **Advantages over pure LR:**

    - More robust for nonlinear systems
    - Converges even when Hessian approximation is poor
    - Uses history to accelerate convergence

    **Advantages over pure AM:**

    - Faster initial convergence (LR provides good initial guess)
    - Fewer iterations typically needed

    **When to Use:**

    - Default choice for most L-FTS simulations
    - Especially good for moderately nonlinear Hamiltonians
    - Recommended when LR alone doesn't converge reliably

    See Also
    --------
    LR : Linear Response compressor alone.
    LFTS.find_saddle_point : Uses this compressor.

    References
    ----------
    .. [1] J. Chem. Phys. 2023, 158, 114117 (Field update methods in FTS)

    Examples
    --------
    >>> # LRAM is typically created inside LFTS.__init__
    >>> from polymerfts import _core
    >>> lr = LR(nx, lx, jk)
    >>> am = factory.create_anderson_mixing(n_var, max_hist, start_error,
    ...                                      mix_min, mix_init)
    >>> lram = LRAM(lr, am)
    >>>
    >>> # In saddle point iteration
    >>> w_new = lram.calculate_new_fields(w_current, -h_deriv, old_err, err)
    """
    def __init__(self, lr, am):
        self.lr = lr
        self.am = am

    def reset_count(self,):
        """Reset Anderson Mixing history.

        Clears the AM history buffer. Should be called when starting a new
        saddle point search (e.g., at beginning of each Langevin step).
        """
        self.am.reset_count()

    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        """Compute new fields using LRAM hybrid method.

        Performs Linear Response followed by Anderson Mixing refinement.

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (1, total_grid) or (total_grid,).
        negative_h_deriv : ndarray
            Negative functional derivative -δH/δw, same shape as w_current.
        old_error_level : float
            Previous error level (used by AM for mixing parameter adjustment).
        error_level : float
            Current error level (used by AM for mixing parameter adjustment).

        Returns
        -------
        w_new : ndarray
            Updated field values after LR + AM, same shape as w_current.

        Notes
        -----
        **Two-Stage Update:**

        1. **Linear Response stage**: Computes w_lr using Hessian inversion
        2. **Anderson Mixing stage**: Refines w_lr using history mixing

        The LR step provides a good initial direction, while AM uses mixing
        history to further accelerate and stabilize convergence.

        **Error Handling:**

        Both old_error_level and error_level are passed to AM for adaptive
        mixing parameter adjustment. LR doesn't use these values.
        """
        nx = self.lr.nx

        w_lr_old = w_current.copy()
        w_lr_new = np.reshape(self.lr.calculate_new_fields(w_lr_old, negative_h_deriv, old_error_level, error_level), [1, np.prod(nx)])
        w_diff = w_lr_new - w_lr_old
        w_new = np.reshape(self.am.calculate_new_fields(w_lr_new, w_diff, old_error_level, error_level), [1, np.prod(nx)])

        return w_new