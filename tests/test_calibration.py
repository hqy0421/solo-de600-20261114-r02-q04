"""Test calibration.py."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pingouin as pg


def _sim(n=150, seed=42, rho=0.0, decreasing=False):
    """Simulate a calibration experiment with magnitude-dependent, possibly
    correlated measurement errors on both instruments."""
    rng = np.random.default_rng(seed)
    xi = np.sort(rng.uniform(0.5, 10, n))
    if decreasing:

        def f(x):
            return 9 - 2.5 * np.log1p(x)

    else:

        def f(x):
            return 1 + 2.5 * np.log1p(x)

    sx = 0.05 + 0.10 * xi / 10  # error of the new instrument grows with magnitude
    sy = 0.05 + 0.03 * np.abs(f(xi))
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x = xi + sx * z1
    y = f(xi) + sy * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    return xi, x, y, sx, sy, f


class TestSplineCalibration:
    def test_forward_recovers_curve(self):
        """The fitted curve recovers the true monotonic curve and is monotone."""
        _, x, y, sx, sy, f = _sim()
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        assert cal.converged_
        assert cal.direction_ == "increasing"
        grid = np.linspace(1.0, 9.5, 50)
        pred = cal.predict(grid)
        # monotonic non-decreasing
        assert np.all(np.diff(pred["y_pred"].to_numpy()) >= 0)
        # close to the true curve
        assert np.abs(pred["y_pred"].to_numpy() - f(grid)).max() < 0.2
        # goodness of fit consistent with the known error model
        assert cal.p_fit_ > 0.01

    def test_eiv_not_ols_coverage(self):
        """Errors-in-variables intervals cover the true curve; treating one
        axis as error-free truth gives overconfident, biased intervals."""
        rng = np.random.default_rng(7)
        n = 200
        xi = np.sort(rng.uniform(0, 10, n))
        x = xi + rng.normal(0, 1.0, n)
        y = xi + rng.normal(0, 0.3, n)  # true curve is the identity
        grid = np.linspace(1, 9, 30)
        cal = pg.spline_calibration(x, y, x_err=1.0, y_err=0.3, n_knots=4)
        pred = cal.predict(grid)
        cov_eiv = np.mean(
            (pred["ci_lower"].to_numpy() <= grid) & (grid <= pred["ci_upper"].to_numpy())
        )
        assert cov_eiv >= 0.8
        # same data, but x (wrongly) treated as error-free truth
        cal_naive = pg.spline_calibration(x, y, x_err=1e-6, y_err=0.3, n_knots=4)
        pred = cal_naive.predict(grid)
        cov_naive = np.mean(
            (pred["ci_lower"].to_numpy() <= grid) & (grid <= pred["ci_upper"].to_numpy())
        )
        assert cov_naive < cov_eiv

    def test_correlated_errors(self):
        """Correlated errors between the two axes are supported; scalar and
        array specification give identical results."""
        _, x, y, sx, sy, f = _sim(rho=0.6, seed=3)
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy, corr=0.6)
        assert cal.converged_
        assert cal.p_fit_ > 0.01
        grid = np.linspace(1.0, 9.5, 50)
        assert np.abs(cal.predict(grid)["y_pred"].to_numpy() - f(grid)).max() < 0.25
        # scalar vs explicit array
        cal_arr = pg.spline_calibration(x, y, x_err=sx, y_err=sy, corr=np.full(x.size, 0.6))
        assert_allclose(cal.coef_, cal_arr.coef_, rtol=1e-8)
        assert_allclose(cal.xi_, cal_arr.xi_, rtol=1e-8)
        # negative correlation
        cal_neg = pg.spline_calibration(x, y, x_err=sx, y_err=sy, corr=-0.4)
        assert cal_neg.converged_

    def test_forward_inverse_roundtrip(self):
        """inverse(predict(x)) == x within the calibration range."""
        _, x, y, sx, sy, _ = _sim()
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        x0 = np.array([1.5, 3.0, 5.0, 7.0, 9.0])
        y0 = cal.predict(x0)["y_pred"].to_numpy()
        x_back = cal.inverse(y0)["x_pred"].to_numpy()
        assert_allclose(x_back, x0, atol=1e-6)

    def test_outside_range_flagged(self):
        """Readings outside the calibration range are clearly flagged."""
        _, x, y, sx, sy, _ = _sim()
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        lo, hi = cal.x_range_
        pred = cal.predict([lo - 5.0, (lo + hi) / 2, hi + 5.0])
        assert pred["outside"].tolist() == [True, False, True]
        # extrapolated values are clipped to the boundary of the curve
        assert_allclose(
            pred["y_pred"].to_numpy()[[0, 2]],
            cal.predict([lo, hi])["y_pred"].to_numpy(),
            rtol=1e-8,
        )
        # inverse: values outside the calibrated reference range
        ylo, yhi = cal.y_range_
        inv = cal.inverse([ylo - 1.0, (ylo + yhi) / 2, yhi + 1.0])
        assert inv["outside"].tolist() == [True, False, True]
        assert inv["x_pred"].to_numpy()[0] == lo
        assert inv["x_pred"].to_numpy()[2] == hi
        # exactly at the boundary is not flagged
        assert not cal.inverse([ylo, yhi])["outside"].any()

    def test_intervals_include_calibration_uncertainty(self):
        """Intervals are centered on the estimate, have positive width, and
        widen when the measurement error of the new observation is added."""
        _, x, y, sx, sy, _ = _sim()
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        x0 = np.array([2.0, 5.0, 8.0])
        pred = cal.predict(x0)
        assert (pred["se"] > 0).all()
        assert (pred["ci_lower"] < pred["y_pred"]).all()
        assert (pred["ci_upper"] > pred["y_pred"]).all()
        pred_e = cal.predict(x0, x_err=0.3)
        assert (pred_e["se"] > pred["se"]).all()
        y0 = pred["y_pred"].to_numpy()
        inv = cal.inverse(y0)
        inv_e = cal.inverse(y0, y_err=0.3)
        assert (inv["se"] > 0).all()
        assert (inv_e["se"] > inv["se"]).all()
        # narrower alpha gives narrower intervals
        pred90 = cal.predict(x0, alpha=0.10)
        assert (
            pred90["ci_upper"].to_numpy() - pred90["ci_lower"].to_numpy()
            < pred["ci_upper"].to_numpy() - pred["ci_lower"].to_numpy()
        ).all()

    def test_decreasing(self):
        """Decreasing curves are detected and handled."""
        _, x, y, sx, sy, f = _sim(decreasing=True, seed=5)
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        assert cal.direction_ == "decreasing"
        grid = np.linspace(1.0, 9.5, 50)
        pred = cal.predict(grid)
        assert np.all(np.diff(pred["y_pred"].to_numpy()) <= 0)
        assert np.abs(pred["y_pred"].to_numpy() - f(grid)).max() < 0.2
        # roundtrip
        y0 = pred["y_pred"].to_numpy()[:10]
        assert_allclose(cal.inverse(y0)["x_pred"].to_numpy(), grid[:10], atol=1e-6)
        # forced direction
        cal_inc = pg.spline_calibration(x, y, x_err=sx, y_err=sy, direction="decreasing")
        assert_allclose(cal_inc.coef_, cal.coef_, rtol=1e-8)

    def test_scalar_array_equivalence(self):
        """Scalar errors equal constant-array errors."""
        _, x, y, _, _, _ = _sim()
        cal_s = pg.spline_calibration(x, y, x_err=0.1, y_err=0.08, corr=0.2)
        cal_a = pg.spline_calibration(
            x,
            y,
            x_err=np.full(x.size, 0.1),
            y_err=np.full(x.size, 0.08),
            corr=np.full(x.size, 0.2),
        )
        assert_allclose(cal_s.coef_, cal_a.coef_, rtol=1e-8)
        assert_allclose(cal_s.cov_coef_, cal_a.cov_coef_, rtol=1e-8)

    def test_knots_and_degree(self):
        """Explicit knots, number of knots and spline degree."""
        _, x, y, sx, sy, _ = _sim()
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy, knots=[3.0, 5.0, 7.0])
        assert_allclose(cal.knots_, [3.0, 5.0, 7.0])
        assert cal.converged_
        for kw in [dict(n_knots=2), dict(degree=2), dict(degree=1), dict(degree=4, n_knots=3)]:
            cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy, **kw)
            assert cal.converged_
            grid = np.linspace(1.0, 9.5, 30)
            assert np.all(np.diff(cal.predict(grid)["y_pred"].to_numpy()) >= 0)

    def test_summary_and_attributes(self):
        """Fitted attributes and coefficient summary."""
        _, x, y, sx, sy, _ = _sim()
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        n = x.size
        assert cal.chi2_ >= 0
        assert cal.dof_ == n - cal.coef_.size
        assert 0 <= cal.p_fit_ <= 1
        assert cal.n_iter_ >= 1
        assert cal.xi_.shape == (n,)
        assert cal.fitted_.shape == (n,)
        assert cal.residuals_.shape == (n, 2)
        assert cal.cov_coef_.shape == (cal.coef_.size, cal.coef_.size)
        assert cal.x_range_[0] < cal.x_range_[1]
        assert cal.y_range_[0] < cal.y_range_[1]
        assert np.all(np.diff(cal.knots_) > 0)
        s = cal.summary()
        assert s.columns.tolist() == ["names", "coef", "se", "CI[2.5%]", "CI[97.5%]"]
        assert len(s) == cal.coef_.size
        assert s["names"].iloc[0] == "Intercept"
        assert (s["se"] >= 0).all()
        # residuals are consistent with the latent values
        assert_allclose(cal.residuals_[:, 0], x - cal.xi_, rtol=1e-8)
        assert_allclose(cal.residuals_[:, 1], y - cal.fitted_, rtol=1e-8)

    def test_no_convergence_warning(self):
        """A warning is emitted when the fit does not converge."""
        _, x, y, sx, sy, _ = _sim()
        with pytest.warns(UserWarning, match="did not converge"):
            cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy, max_iter=1)
        assert not cal.converged_

    def test_constant_y(self):
        """A constant reference gives a flat curve without warnings."""
        rng = np.random.default_rng(11)
        x = np.sort(rng.uniform(0, 10, 30))
        y = 3.0 + rng.normal(0, 0.05, x.size)
        with warnings.catch_warnings():
            # no warning (in particular no invalid-correlation warning)
            warnings.simplefilter("error")
            cal = pg.spline_calibration(x, y, x_err=0.1, y_err=0.05)
        pred = cal.predict([2.0, 8.0])
        assert_allclose(pred["y_pred"].to_numpy(), 3.0, atol=0.05)

    def test_exported(self):
        """The function and the fit class are exposed at the top level."""
        _, x, y, sx, sy, _ = _sim()
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        assert isinstance(cal, pg.SplineCalibration)

    def test_input_validation(self):
        """Invalid inputs raise informative errors."""
        _, x, y, sx, sy, _ = _sim()
        # error correlation outside (-1, 1)
        with pytest.raises(ValueError, match="corr"):
            pg.spline_calibration(x, y, x_err=sx, y_err=sy, corr=1.0)
        with pytest.raises(ValueError, match="corr"):
            pg.spline_calibration(x, y, x_err=sx, y_err=sy, corr=-1.3)
        # non-positive standard errors
        with pytest.raises(ValueError, match="x_err"):
            pg.spline_calibration(x, y, x_err=0.0, y_err=sy)
        with pytest.raises(ValueError, match="y_err"):
            pg.spline_calibration(x, y, x_err=sx, y_err=-1.0)
        # wrong length
        with pytest.raises(ValueError, match="x_err"):
            pg.spline_calibration(x, y, x_err=sx[:-1], y_err=sy)
        with pytest.raises(ValueError, match="same length"):
            pg.spline_calibration(x[:-1], y, x_err=sx[:-1], y_err=sy)
        # NaN / inf
        with pytest.raises(ValueError, match="NaN"):
            pg.spline_calibration(np.where(x > 5, np.nan, x), y, x_err=sx, y_err=sy)
        with pytest.raises(ValueError, match="NaN"):
            pg.spline_calibration(x, y, x_err=sx, y_err=np.inf)
        # constant x
        with pytest.raises(ValueError, match="distinct"):
            pg.spline_calibration(np.ones(10), y[:10], x_err=0.1, y_err=0.1)
        # invalid direction / degree / knots / max_iter
        with pytest.raises(ValueError, match="direction"):
            pg.spline_calibration(x, y, x_err=sx, y_err=sy, direction="up")
        with pytest.raises(ValueError, match="degree"):
            pg.spline_calibration(x, y, x_err=sx, y_err=sy, degree=0)
        with pytest.raises(ValueError, match="knots"):
            pg.spline_calibration(x, y, x_err=sx, y_err=sy, knots=[-1.0, 5.0])
        with pytest.raises(ValueError, match="max_iter"):
            pg.spline_calibration(x, y, x_err=sx, y_err=sy, max_iter=0)
        # too few observations for the spline
        with pytest.raises(ValueError, match="Not enough"):
            pg.spline_calibration(x[:5], y[:5], x_err=sx[:5], y_err=sy[:5])
        # invalid alpha / non-finite new values in predict and inverse
        cal = pg.spline_calibration(x, y, x_err=sx, y_err=sy)
        with pytest.raises(ValueError, match="alpha"):
            cal.predict([1.0], alpha=1.5)
        with pytest.raises(ValueError, match="NaN"):
            cal.predict([np.nan])
        with pytest.raises(ValueError, match="NaN"):
            cal.inverse([np.inf])
        with pytest.raises(ValueError, match="x_err"):
            cal.predict([1.0], x_err=-0.1)
