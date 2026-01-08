"""ADAM optimizer for SCFT field optimization."""

import numpy as np

# For ADAM optimizer, see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
class Adam:
    """ADAM optimizer for SCFT field optimization.

    Implements the ADAM (Adaptive Moment Estimation) optimization algorithm
    for finding saddle points in self-consistent field theory calculations.
    This is an alternative to Anderson Mixing for field optimization.

    The algorithm maintains first and second moment estimates of gradients
    and uses them to adapt the learning rate for each parameter. This can
    provide more stable convergence for some systems.

    Parameters
    ----------
    total_grid : int
        Total number of grid points (size of field arrays to optimize).
    lr : float, optional
        Initial learning rate, γ (default: 1e-2). Controls step size.
    b1 : float, optional
        Exponential decay rate for first moment estimates, β₁ (default: 0.9).
        Should be in [0, 1).
    b2 : float, optional
        Exponential decay rate for second moment estimates, β₂ (default: 0.999).
        Should be in [0, 1).
    eps : float, optional
        Small constant for numerical stability, ε (default: 1e-8).
        Prevents division by zero.
    gamma : float, optional
        Learning rate decay factor (default: 1.0).
        Learning rate at iteration T is lr * γ^(T-1).

    Attributes
    ----------
    total_grid : int
        Size of field arrays.
    lr : float
        Base learning rate.
    b1, b2 : float
        Moment decay rates.
    eps : float
        Numerical stability constant.
    gamma : float
        Learning rate decay factor.
    count : int
        Current iteration number.
    m : ndarray
        First moment estimates (moving average of gradients).
    v : ndarray
        Second moment estimates (moving average of squared gradients).

    See Also
    --------
    SCFT : Main SCFT class that uses this optimizer.

    Notes
    -----
    This implementation follows the ADAM algorithm from [1]_. The update rule is:

    .. math::
        m_t = β_1 m_{t-1} + (1-β_1) g_t

        v_t = β_2 v_{t-1} + (1-β_2) g_t^2

        \\hat{m}_t = m_t / (1 - β_1^t)

        \\hat{v}_t = v_t / (1 - β_2^t)

        w_t = w_{t-1} + γ_t \\hat{m}_t / (\\sqrt{\\hat{v}_t} + ε)

    where g_t is the gradient (field difference) at iteration t.

    References
    ----------
    .. [1] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
           optimization. ICLR 2015.

    .. [2] https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    Examples
    --------
    >>> # Create ADAM optimizer for 32x32x32 grid
    >>> optimizer = Adam(total_grid=32*32*32, lr=1e-2)
    >>>
    >>> # In SCFT iteration loop
    >>> w_new = optimizer.calculate_new_fields(w_current, w_diff,
    ...                                        old_error, error_level)
    """
    def __init__(self, total_grid,
                    lr = 1e-2,       # initial learning rate, γ
                    b1 = 0.9,        # β1
                    b2 = 0.999,      # β2
                    eps = 1e-8,      # epsilon, small number to prevent dividing by zero
                    gamma = 1.0,     # learning rate at Tth iteration is lr*γ^(T-1)
                    ):
        self.total_grid = total_grid
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.gamma = gamma
        self.count = 1

        self.m = np.zeros(total_grid, dtype=np.float64) # first moment
        self.v = np.zeros(total_grid, dtype=np.float64) # second moment

    def reset_count(self,):
        """Reset optimizer state for a new optimization run.

        Resets the iteration counter to 1 and clears the moment estimates.
        This should be called when starting a new SCFT run with different
        initial conditions.

        Notes
        -----
        This method reinitializes:
        - Iteration counter (count) to 1
        - First moment estimate (m) to zeros
        - Second moment estimate (v) to zeros
        """
        self.count = 1
        self.m[:] = 0.0
        self.v[:] = 0.0

    def calculate_new_fields(self, w_current, w_diff, old_error_level, error_level):
        """Compute updated fields using ADAM optimization algorithm.

        Performs one step of ADAM optimization to update the field values
        based on the current fields and their gradients (field differences).

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (total_grid,) or (M, grid_size).
        w_diff : ndarray
            Field gradients (self-consistency error), same shape as w_current.
        old_error_level : float
            Error level from previous iteration (not used in ADAM, kept for
            compatibility with Anderson Mixing interface).
        error_level : float
            Current error level (not used in ADAM, kept for compatibility).

        Returns
        -------
        w_new : ndarray
            Updated field values, same shape as w_current.

        Notes
        -----
        The ADAM update includes:

        1. Decay learning rate: lr_t = lr * γ^(t-1)
        2. Update biased first moment: m_t = β₁ m_{t-1} + (1-β₁) g_t
        3. Update biased second moment: v_t = β₂ v_{t-1} + (1-β₂) g_t²
        4. Bias correction: m̂_t = m_t/(1-β₁^t), v̂_t = v_t/(1-β₂^t)
        5. Update fields: w_t = w_{t-1} + lr_t * m̂_t / (√v̂_t + ε)

        The bias correction in steps 4 compensates for the initialization
        of moment estimates at zero, particularly important in early iterations.
        """
        lr = self.lr*self.gamma**(self.count-1)

        self.m = self.b1*self.m + (1.0-self.b1)*w_diff
        self.v = self.b2*self.v + (1.0-self.b2)*w_diff**2
        m_hat = self.m/(1.0-self.b1**self.count)
        v_hat = self.v/(1.0-self.b2**self.count)

        w_new = w_current + lr*m_hat/(np.sqrt(v_hat) + self.eps)

        self.count += 1
        return w_new
