"""Monotonic spline calibration with errors in both variables."""

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import pinvh
from scipy.optimize import brentq, lsq_linear, minimize
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import norm

__all__ = ["spline_calibration", "SplineCalibration"]


class _ISplineBasis:
    """Monotone I-spline basis on a fixed interval.

    I-splines (Ramsay, 1988) are integrals of normalized B-splines
    (M-splines). Each basis function is non-decreasing and rises from 0 to 1
    over the interval, so a linear combination with non-negative coefficients
    is a non-decreasing spline. Values are evaluated with clipping, i.e. the
    basis is constant outside ``[a, b]``.
    """

    def __init__(self, a, b, interior, degree):
        d = degree - 1  # degree of the underlying M-splines
        t = np.concatenate(([a] * (d + 1), interior, [b] * (d + 1)))
        m = len(t) - d - 1
        # M-spline normalization: each basis function integrates to one
        scale = (d + 1) / (t[d + 1 : d + 1 + m] - t[:m])
        mspline = BSpline(t, np.diag(scale), d)
        self._ispline = mspline.antiderivative()
        self._mspline = mspline
        self.a, self.b, self.m, self.degree = a, b, m, degree

    def design(self, x):
        """Design matrix (intercept + I-splines), shape (n, m + 1)."""
        x = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), self.a, self.b)
        return np.column_stack([np.ones_like(x), self._ispline(x)])

    def deriv(self, x):
        """Derivative (M-spline) basis, shape (n, m)."""
        x = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), self.a, self.b)
        return self._mspline(x)


def _fit_eiv(x, y, sx, sy, c, basis, max_iter, tol):
    """Alternating maximum-likelihood fit of a monotone spline with errors in
    both variables.

    Minimizes, over the spline coefficients (constrained to give a
    non-decreasing curve) and the latent true values ``xi_i``, the generalized
    sum of squares

        S = sum_i [x_i - xi_i, y_i - f(xi_i)] W_i [x_i - xi_i, y_i - f(xi_i)]^T

    where ``W_i`` is the inverse of the known 2 x 2 error covariance matrix of
    observation ``i`` (York et al., 2004, generalized to a spline mean).

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
        Observed readings on both instruments.
    sx, sy : np.ndarray, shape (n,)
        Standard measurement errors of ``x`` and ``y``.
    c : np.ndarray, shape (n,)
        Covariance between the errors of ``x`` and ``y``.
    basis : _ISplineBasis
    max_iter, tol : int, float

    Returns
    -------
    beta : np.ndarray, shape (m + 1,)
    xi : np.ndarray, shape (n,)
    S : float
        Value of the generalized sum of squares at convergence.
    n_iter : int
    converged : bool
    """
    n = x.size
    sx2, sy2 = sx**2, sy**2
    det = sx2 * sy2 - c**2
    wxx = sy2 / det
    wxy = -c / det
    wyy = sx2 / det
    sw = np.sqrt(wyy)
    # Conditional adjustment of the y-target for correlated errors:
    # E[y_i | x_i, xi_i] = f(xi_i) + (c_i / sx_i^2) * (x_i - xi_i)
    adj = c / sx2

    lb = np.full(basis.m + 1, -np.inf)
    lb[1:] = 0.0  # non-negative I-spline coefficients => non-decreasing curve
    ub = np.full(basis.m + 1, np.inf)

    def solve_beta(xi):
        A = basis.design(xi)
        res = lsq_linear(A * sw[:, None], (y - adj * (x - xi)) * sw, bounds=(lb, ub))
        return res.x

    def solve_xi(xi, beta):
        def obj(xi, beta):
            f = basis.design(xi) @ beta
            fp = basis.deriv(xi) @ beta[1:]
            dx, dy = x - xi, y - f
            S = np.sum(wxx * dx**2 + 2 * wxy * dx * dy + wyy * dy**2)
            grad = -2 * wxx * dx - 2 * wxy * (dy + dx * fp) - 2 * wyy * fp * dy
            return S, grad

        res = minimize(
            obj,
            xi,
            args=(beta,),
            jac=True,
            method="L-BFGS-B",
            bounds=[(basis.a, basis.b)] * n,
            options={"maxiter": 100},
        )
        return res.x, res.fun

    beta = solve_beta(x)
    xi, S = solve_xi(x, beta)
    n_iter = 0
    converged = False
    while n_iter < max_iter and not converged:
        n_iter += 1
        beta = solve_beta(xi)
        xi, S_new = solve_xi(xi, beta)
        converged = abs(S - S_new) <= tol * (1 + S)
        S = S_new
    return beta, xi, S, n_iter, converged


class SplineCalibration:
    """Fitted monotonic spline calibration curve.

    This class is not meant to be instantiated directly; it is returned by
    :py:func:`pingouin.spline_calibration`.

    Attributes
    ----------
    coef_ : np.ndarray
        Fitted coefficients (intercept first, then the non-negative I-spline
        coefficients), on the internal non-decreasing scale.
    cov_coef_ : np.ndarray
        Approximate covariance matrix of the coefficients, used to propagate
        the calibration uncertainty in :py:meth:`predict` and
        :py:meth:`inverse`.
    xi_ : np.ndarray
        Estimated latent true values of the new-instrument readings.
    fitted_ : np.ndarray
        Fitted values of the reference readings, i.e. the curve evaluated at
        ``xi_`` (original scale).
    residuals_ : np.ndarray
        Array of shape (n, 2) with the residuals of ``x`` (``x - xi_``) and of
        ``y`` (``y - fitted_``).
    knots_ : np.ndarray
        Interior knots of the spline.
    degree_ : int
        Degree of the spline.
    direction_ : str
        ``'increasing'`` or ``'decreasing'``.
    chi2_ : float
        Generalized (Mahalanobis) sum of squares at the optimum. If the error
        model and the spline are correctly specified, this approximately
        follows a chi-square distribution with ``dof_`` degrees of freedom.
    dof_ : int
        Degrees of freedom of ``chi2_`` (``n - n_params``).
    p_fit_ : float
        P-value of the goodness-of-fit test (small values indicate a
        misspecified curve or underestimated measurement errors).
    n_iter_ : int
        Number of alternating-optimization iterations.
    converged_ : bool
        Whether the alternating optimization converged within ``max_iter``.
    x_range_ : tuple
        Calibration range on the new-instrument scale (min, max of ``x``).
    y_range_ : tuple
        Calibration range on the reference scale (image of ``x_range_``).
    n_ : int
        Number of calibration observations.
    """

    def __init__(
        self,
        *,
        basis,
        coef,
        cov_coef,
        xi,
        sign,
        direction,
        chi2,
        dof,
        n_iter,
        converged,
        knots,
    ):
        self._basis = basis
        self._sign = sign
        self.coef_ = coef
        self.cov_coef_ = cov_coef
        self.xi_ = xi
        self.knots_ = knots
        self.degree_ = basis.degree
        self.direction_ = direction
        self.chi2_ = chi2
        self.dof_ = dof
        self.p_fit_ = float(_chi2_dist.sf(chi2, dof)) if dof > 0 else np.nan
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.n_ = xi.size
        self.x_range_ = (basis.a, basis.b)
        glo, ghi = basis.design([basis.a, basis.b]) @ coef
        self._glo, self._ghi = glo, ghi
        yr = sorted([sign * glo, sign * ghi])
        self.y_range_ = (yr[0], yr[1])
        self.fitted_ = sign * (basis.design(xi) @ coef)

    def __repr__(self):
        return (
            f"<SplineCalibration n={self.n_}, direction='{self.direction_}', "
            f"degree={self.degree_}, n_knots={self.knots_.size}, "
            f"chi2/dof={self.chi2_:.1f}/{self.dof_}, converged={self.converged_}>"
        )

    def _check_alpha(self, alpha):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be strictly between 0 and 1.")
        return norm.ppf(1 - alpha / 2)

    def _curve_minus(self, u, y):
        """Fitted curve evaluated at u (internal scale), minus y."""
        return float((self._basis.design([u]) @ self.coef_)[0] - y)

    def predict(self, x_new, x_err=None, alpha=0.05):
        """Convert new-instrument readings to the reference scale (forward).

        Parameters
        ----------
        x_new : float or array_like
            New-instrument readings to convert.
        x_err : float or array_like, optional
            Standard measurement error of the new readings. If given, its
            contribution (propagated through the local slope of the curve) is
            added to the calibration uncertainty.
        alpha : float
            Significance level of the confidence intervals. Default 0.05.

        Returns
        -------
        pd.DataFrame
            One row per reading, with columns:

            - ``x``: input reading
            - ``y_pred``: converted value on the reference scale
            - ``se``: standard error (calibration uncertainty, plus the
              propagated ``x_err`` if given)
            - ``ci_lower``, ``ci_upper``: bounds of the
              ``100 * (1 - alpha)``% confidence interval
            - ``outside``: True if the reading lies outside the calibration
              range ``x_range_``; the value is then an extrapolation computed
              from the nearest boundary of the curve and should not be trusted.
        """
        z = self._check_alpha(alpha)
        x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
        if not np.all(np.isfinite(x_new)):
            raise ValueError("x_new must not contain NaN or infinite values.")
        A = self._basis.design(x_new)
        g = A @ self.coef_
        var = np.einsum("ij,jk,ik->i", A, self.cov_coef_, A)
        if x_err is not None:
            xe = np.broadcast_to(np.asarray(x_err, dtype=float), x_new.shape)
            if not np.all(np.isfinite(xe)) or np.any(xe < 0):
                raise ValueError("x_err must be finite and non-negative.")
            fp = self._basis.deriv(x_new) @ self.coef_[1:]
            var = var + (fp * xe) ** 2
        se = np.sqrt(np.maximum(var, 0.0))
        y_pred = self._sign * g
        outside = (x_new < self.x_range_[0]) | (x_new > self.x_range_[1])
        return pd.DataFrame(
            {
                "x": x_new,
                "y_pred": y_pred,
                "se": se,
                "ci_lower": y_pred - z * se,
                "ci_upper": y_pred + z * se,
                "outside": outside,
            }
        )

    def inverse(self, y_new, y_err=None, alpha=0.05):
        """Convert reference-scale values to the new-instrument scale (inverse).

        Parameters
        ----------
        y_new : float or array_like
            Reference-scale values to convert back.
        y_err : float or array_like, optional
            Standard measurement error of the reference values. If given, its
            contribution (propagated through the local slope of the curve) is
            added to the calibration uncertainty.
        alpha : float
            Significance level of the confidence intervals. Default 0.05.

        Returns
        -------
        pd.DataFrame
            One row per value, with columns:

            - ``y``: input value
            - ``x_pred``: converted value on the new-instrument scale
            - ``se``: standard error (delta method; infinite where the curve
              is locally flat, i.e. where the inverse is not identifiable)
            - ``ci_lower``, ``ci_upper``: bounds of the
              ``100 * (1 - alpha)``% confidence interval
            - ``outside``: True if the value lies outside the calibrated
              reference range ``y_range_``; ``x_pred`` is then the nearest
              boundary of the calibration range and should not be trusted.
        """
        z = self._check_alpha(alpha)
        y_new = np.atleast_1d(np.asarray(y_new, dtype=float))
        if not np.all(np.isfinite(y_new)):
            raise ValueError("y_new must not contain NaN or infinite values.")
        yt = self._sign * y_new  # internal non-decreasing scale
        a, b = self.x_range_
        outside = (yt < self._glo) | (yt > self._ghi)
        yc = np.clip(yt, self._glo, self._ghi)
        x_hat = np.empty_like(yc)
        for i, yi in enumerate(yc):
            if yi == self._glo:
                x_hat[i] = a
            elif yi == self._ghi:
                x_hat[i] = b
            else:
                x_hat[i] = brentq(self._curve_minus, a, b, args=(yi,), xtol=1e-10, rtol=1e-12)
        A = self._basis.design(x_hat)
        var = np.einsum("ij,jk,ik->i", A, self.cov_coef_, A)
        if y_err is not None:
            ye = np.broadcast_to(np.asarray(y_err, dtype=float), y_new.shape)
            if not np.all(np.isfinite(ye)) or np.any(ye < 0):
                raise ValueError("y_err must be finite and non-negative.")
            var = var + ye**2
        fp = self._basis.deriv(x_hat) @ self.coef_[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.sqrt(np.maximum(var, 0.0)) / np.abs(fp)
        se[np.abs(fp) < 1e-12] = np.inf
        return pd.DataFrame(
            {
                "y": y_new,
                "x_pred": x_hat,
                "se": se,
                "ci_lower": x_hat - z * se,
                "ci_upper": x_hat + z * se,
                "outside": outside,
            }
        )

    def summary(self, alpha=0.05):
        """Summary of the fitted spline coefficients.

        Parameters
        ----------
        alpha : float
            Significance level of the confidence intervals. Default 0.05.

        Returns
        -------
        pd.DataFrame
            Coefficients (intercept and I-spline basis functions, on the
            internal non-decreasing scale) with their standard errors and
            confidence intervals. No p-values are reported: the non-negativity
            constraints on the spline coefficients make classical Wald tests
            invalid.
        """
        z = self._check_alpha(alpha)
        se = np.sqrt(np.maximum(np.diag(self.cov_coef_), 0.0))
        names = ["Intercept"] + [f"I{j}" for j in range(1, self.coef_.size)]
        return pd.DataFrame(
            {
                "names": names,
                "coef": self.coef_,
                "se": se,
                "CI[%.1f%%]" % (100 * alpha / 2): self.coef_ - z * se,
                "CI[%.1f%%]" % (100 * (1 - alpha / 2)): self.coef_ + z * se,
            }
        )


def spline_calibration(
    x,
    y,
    x_err,
    y_err,
    corr=0.0,
    degree=3,
    n_knots=None,
    knots=None,
    direction="auto",
    max_iter=500,
    tol=1e-8,
):
    """Monotonic spline calibration with errors in both variables.

    Fit a monotonic spline calibration curve that converts the readings of a
    new instrument to a reference scale, when **both** instruments have known
    measurement errors that may vary in magnitude from point to point and may
    be correlated with each other (errors-in-variables model).

    .. versionadded:: 0.7.0

    Parameters
    ----------
    x : array_like
        Readings of the new instrument, shape *(n,)*.
    y : array_like
        Readings of the reference instrument, shape *(n,)*.
    x_err : float or array_like
        Known standard measurement error of ``x``: a single value, or one
        value per observation (e.g. proportional to the magnitude).
    y_err : float or array_like
        Known standard measurement error of ``y``: a single value, or one
        value per observation.
    corr : float or array_like
        Correlation between the measurement errors of ``x`` and ``y``, e.g.
        when both instruments respond to a common uncontrolled factor. A
        single value or one value per observation. Must lie in (-1, 1).
        Default 0 (independent errors).
    degree : int
        Degree of the monotonic spline. Default 3 (cubic).
    n_knots : int, optional
        Number of interior knots, placed at quantiles of ``x``. If None
        (default), chosen as ``max(1, min(5, (n - 2) // 10))``.
    knots : array_like, optional
        Explicit interior knots (strictly inside the range of ``x``).
        Overrides ``n_knots``.
    direction : str
        Expected monotonic direction of the curve: ``'auto'`` (default,
        inferred from the sign of the correlation between ``x`` and ``y``),
        ``'increasing'`` or ``'decreasing'``.
    max_iter : int
        Maximum number of alternating-optimization iterations. Default 500.
    tol : float
        Relative tolerance on the objective for convergence. Default 1e-8.

    Returns
    -------
    cal : SplineCalibration
        Fitted calibration object. Use :py:meth:`SplineCalibration.predict`
        for forward conversion (new instrument -> reference) and
        :py:meth:`SplineCalibration.inverse` for inverse conversion
        (reference -> new instrument); both return confidence intervals that
        include the calibration uncertainty and flag out-of-range readings.

    See Also
    --------
    linear_regression : (Multiple) linear regression.
    logistic_regression : Logistic regression.

    Notes
    -----
    **Errors-in-variables model.** Ordinary least squares treats one axis
    (typically the reference) as error-free truth, which biases the curve and
    its uncertainty whenever both instruments are noisy. Here, both axes are
    modeled as noisy observations of latent true values:

    .. math::

        x_i = \\xi_i + e_i, \\qquad y_i = f(\\xi_i) + u_i,

    .. math::

        (e_i, u_i) \\sim \\mathcal{N}(0, \\Sigma_i), \\qquad
        \\Sigma_i = \\begin{pmatrix}
        \\sigma_{x,i}^2 & \\rho_i \\sigma_{x,i} \\sigma_{y,i} \\\\
        \\rho_i \\sigma_{x,i} \\sigma_{y,i} & \\sigma_{y,i}^2
        \\end{pmatrix}

    with known :math:`\\Sigma_i`. The curve :math:`f` and the latent values
    :math:`\\xi_i` are estimated by maximum likelihood, i.e. by minimizing the
    generalized (Mahalanobis) sum of squares

    .. math::

        S = \\sum_i \\begin{pmatrix} x_i - \\xi_i \\\\ y_i - f(\\xi_i)
        \\end{pmatrix}^{\\!T} \\Sigma_i^{-1}
        \\begin{pmatrix} x_i - \\xi_i \\\\ y_i - f(\\xi_i) \\end{pmatrix},

    which generalizes the York et al. (2004) bivariate line fit to a spline
    mean. The optimization alternates between the coefficients (a weighted,
    sign-constrained least-squares problem) and the latent values (a bound-
    constrained smooth problem), and monotonically decreases :math:`S`.

    **Monotonicity.** :math:`f` is an I-spline (Ramsay, 1988): a linear
    combination of integrated normalized B-splines with non-negative
    coefficients, which guarantees a monotonic curve without any further
    constraint on the shape.

    **Uncertainty.** Coefficient uncertainty is approximated by the
    covariance matrix of the weighted fit at convergence, using the effective
    variance :math:`\\sigma_{y,i}^2 + f'(\\hat\\xi_i)^2 \\sigma_{x,i}^2
    - 2 f'(\\hat\\xi_i) \\, \\text{Cov}_i` of each point (the errors are
    assumed known, so no residual rescaling is applied). Forward intervals
    are Wald intervals on the curve; inverse intervals use the delta method
    through the local slope and become wide where the curve is flat. Both
    include the calibration uncertainty; the measurement error of a new
    observation can be added via ``x_err`` / ``y_err`` in
    :py:meth:`SplineCalibration.predict` and
    :py:meth:`SplineCalibration.inverse`.

    **Extrapolation.** The curve is only identified on the calibration range
    (``x_range_`` on the new-instrument scale, ``y_range_`` on the reference
    scale). Conversions outside these ranges are clipped to the nearest
    boundary and flagged with ``outside=True``; do not use them.

    **Goodness of fit.** If the error model is correct, ``chi2_``
    approximately follows a chi-square distribution with ``dof_`` degrees of
    freedom; a small ``p_fit_`` indicates a misspecified curve or
    underestimated measurement errors.

    References
    ----------
    .. [1] Ramsay, J. O. (1988). Monotone regression splines in action.
       *Statistical Science*, 3(4), 425-441.

    .. [2] York, D., Evensen, N. M., Martinez, M. L., & Delgado, J. D. B.
       (2004). Unified equations for the slope, intercept, and standard
       errors of the best straight line. *American Journal of Physics*,
       72(3), 367-375.

    .. [3] EURACHEM/CITAC Guide CG 4 (2012). *Quantifying uncertainty in
       analytical measurement* (3rd ed.), chapter 5.

    Examples
    --------
    >>> import numpy as np
    >>> import pingouin as pg
    >>> rng = np.random.default_rng(42)
    >>> xi = np.linspace(1, 10, 40)                       # latent true values
    >>> f = lambda x: 2 + 3 * np.log1p(x)                 # true curve
    >>> x = xi + rng.normal(0, 0.20, xi.size)             # new instrument
    >>> y = f(xi) + rng.normal(0, 0.15, xi.size)          # reference instrument
    >>> cal = pg.spline_calibration(x, y, x_err=0.20, y_err=0.15)
    >>> print(cal.direction_, cal.converged_, round(cal.p_fit_, 3))
    increasing True 0.882

    Forward conversion (new instrument -> reference). The last reading is
    outside the calibration range and is flagged:

    >>> cal.predict([2.0, 5.0, 30.0]).round(3)
          x  y_pred    se  ci_lower  ci_upper  outside
    0   2.0   5.505 0.110     5.289     5.720    False
    1   5.0   7.373 0.061     7.253     7.493    False
    2  30.0   9.206 0.146     8.920     9.493     True

    Inverse conversion (reference -> new instrument):

    >>> cal.inverse([5.0, 8.0]).round(3)
         y  x_pred    se  ci_lower  ci_upper  outside
    0  5.0   1.575 0.087     1.405     1.745    False
    1  8.0   6.345 0.165     6.023     6.668    False
    """
    # ---- Input validation
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be one-dimensional arrays of the same length.")
    n = x.size
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must not contain NaN or infinite values.")

    def _broadcast_err(v, name, positive):
        try:
            v = np.broadcast_to(np.asarray(v, dtype=float), (n,)).astype(float)
        except ValueError:
            raise ValueError(f"{name} must be a scalar or have the same length as x.") from None
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} must not contain NaN or infinite values.")
        if positive and np.any(v <= 0):
            raise ValueError(f"{name} must be strictly positive.")
        if not positive and np.any(np.abs(v) >= 1):
            raise ValueError(f"{name} must lie strictly between -1 and 1.")
        return v

    sx = _broadcast_err(x_err, "x_err", positive=True)
    sy = _broadcast_err(y_err, "y_err", positive=True)
    rho = _broadcast_err(corr, "corr", positive=False)

    a, b = float(x.min()), float(x.max())
    if not a < b:
        raise ValueError("x must take at least two distinct values.")

    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be a positive integer.")
    if knots is not None:
        interior = np.unique(np.asarray(knots, dtype=float))
        if interior.ndim != 1 or not np.all(np.isfinite(interior)):
            raise ValueError("knots must be a finite one-dimensional array.")
        if np.any((interior <= a) | (interior >= b)):
            raise ValueError("knots must lie strictly inside the range of x.")
    else:
        if n_knots is None:
            n_knots = max(1, min(5, (n - 2) // 10))
        n_knots = int(n_knots)
        if n_knots < 0:
            raise ValueError("n_knots must be a non-negative integer.")
        probs = np.linspace(0, 1, n_knots + 2)[1:-1]
        interior = np.quantile(x, probs) if probs.size else np.array([])
        interior = np.unique(interior[(interior > a) & (interior < b)])

    direction = str(direction).lower()
    if direction == "auto":
        with np.errstate(invalid="ignore"):
            r = np.corrcoef(x, y)[0, 1]
        sign = 1.0 if not np.isfinite(r) or r >= 0 else -1.0
        direction = "increasing" if sign > 0 else "decreasing"
    elif direction == "increasing":
        sign = 1.0
    elif direction == "decreasing":
        sign = -1.0
    else:
        raise ValueError("direction must be 'auto', 'increasing' or 'decreasing'.")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if tol <= 0:
        raise ValueError("tol must be strictly positive.")

    basis = _ISplineBasis(a, b, interior, degree)
    if n <= basis.m + 1:
        raise ValueError(
            f"Not enough observations ({n}) for {basis.m + 1} spline coefficients; "
            "use fewer knots or a lower degree."
        )

    # ---- Fit on the internal non-decreasing scale (y -> sign * y)
    yt = sign * y
    ct = sign * rho * sx * sy  # error covariance of (x, sign * y)
    beta, xi, S, n_iter, converged = _fit_eiv(x, yt, sx, sy, ct, basis, max_iter, tol)
    if not converged:
        warnings.warn(
            f"spline_calibration did not converge within max_iter={max_iter} "
            "iterations; consider increasing max_iter.",
            UserWarning,
            stacklevel=2,
        )

    # ---- Approximate covariance of the coefficients (York effective variance)
    fp = basis.deriv(xi) @ beta[1:]
    var_eff = np.maximum(sy**2 + (fp * sx) ** 2 - 2 * fp * ct, 1e-14)
    A = basis.design(xi)
    cov_coef = pinvh(A.T @ (A / var_eff[:, None]))

    cal = SplineCalibration(
        basis=basis,
        coef=beta,
        cov_coef=cov_coef,
        xi=xi,
        sign=sign,
        direction=direction,
        chi2=float(S),
        dof=n - basis.m - 1,
        n_iter=n_iter,
        converged=converged,
        knots=interior,
    )
    cal.residuals_ = np.column_stack([x - xi, y - cal.fitted_])
    return cal
