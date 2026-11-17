"""Analytic checks of Fourier conventions and field operations."""

import operator

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spectrumflow import ToroidalField


@pytest.fixture
def field():
    return ToroidalField.from_modes(
        {(0, 0): 2, (0, 1): 0.3j, (1, -2): 0.2 - 0.4j, (2, 1): 0.5 + 0.1j},
        nfp=3,
    )


def test_constant_zero_mode_and_broadcasting():
    f = ToroidalField.constant(3.5, nfp=2, mpol=2, ntor=1)
    assert f.mean == 3.5
    assert f(0, 0).shape == ()
    assert_allclose(f(0, 0), 3.5, rtol=0, atol=1e-14)
    values = f(np.zeros((2, 1, 1)), np.zeros((1, 3, 4)))
    assert values.shape == (2, 3, 4)
    assert_allclose(values, 3.5, rtol=0, atol=1e-14)
    assert_allclose(f.derivative(1).coefficients, 0, rtol=0, atol=1e-14)
    assert_allclose(f.derivative(zeta_order=2).coefficients, 0, rtol=0, atol=1e-14)
    assert f([], []).shape == (0,)


@pytest.mark.parametrize("m,n", [(0, 2), (1, -2), (2, 1), (2, 0)])
def test_single_mode_value_derivatives_and_periodicity(m, n):
    f = ToroidalField.from_modes({(0, 0): 1.2, (m, n): (2 - 3j) / 2}, nfp=4)
    theta = np.array([0.13, 0.82, 2.1])[:, None]
    zeta = np.array([0.04, 0.27])[None, :]
    phase = m * theta - n * 4 * zeta
    oscillation = 2 * np.cos(phase) + 3 * np.sin(phase)
    slope = -2 * np.sin(phase) + 3 * np.cos(phase)
    assert_allclose(f(theta, zeta), 1.2 + oscillation, rtol=1e-13, atol=1e-13)
    assert_allclose(f.derivative(1)(theta, zeta), m * slope, rtol=1e-13, atol=1e-13)
    assert_allclose(f.derivative(zeta_order=1)(theta, zeta), -4*n*slope, rtol=1e-13, atol=1e-13)
    assert_allclose(f.derivative(1, 1)(theta, zeta), 4*m*n*oscillation, rtol=1e-13, atol=1e-13)
    assert_allclose(f.derivative(2)(theta, zeta), -m*m*oscillation, rtol=1e-13, atol=1e-13)
    assert_allclose(f(theta + 2*np.pi, zeta + 2*np.pi/4), f(theta, zeta), rtol=1e-13, atol=1e-13)
    assert f.coefficient(-m, -n) == f.coefficient(m, n).conjugate()


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 1), (5, 7)])
def test_fft_round_trip_for_arbitrary_real_data(shape):
    values = np.random.default_rng(31).normal(size=shape)
    f = ToroidalField.from_samples(values, nfp=5)
    theta, zeta, restored = f.sample()
    assert restored.shape == shape
    assert_allclose(theta, 2*np.pi*np.arange(shape[0])/shape[0], rtol=0, atol=1e-14)
    assert_allclose(zeta, 2*np.pi*np.arange(shape[1])/(5*shape[1]), rtol=0, atol=1e-14)
    assert_allclose(restored, values, rtol=1e-13, atol=1e-13)
    assert_allclose(f.mean, values.mean(), rtol=1e-13, atol=1e-13)


def test_fft_sign_normalization_and_off_grid_evaluation():
    theta = 2*np.pi*np.arange(7)/7
    zeta = 2*np.pi*np.arange(9)/(3*9)

    def analytic(t, z):
        return 2 + 0.8*np.cos(2*t - 6*z) + 0.6*np.sin(t + 3*z) + 0.4*np.sin(3*z)

    f = ToroidalField.from_samples(analytic(theta[:, None], zeta[None, :]), nfp=3)
    assert_allclose(f.coefficient(2, 2), 0.4, rtol=0, atol=1e-14)
    assert_allclose(f.coefficient(1, -1), -0.3j, rtol=0, atol=1e-14)
    assert_allclose(f.coefficient(0, 1), 0.2j, rtol=0, atol=1e-14)
    assert_allclose(f(0.123, 0.456), analytic(0.123, 0.456), rtol=1e-13, atol=1e-13)


def test_arithmetic_resolution_and_no_aliasing(field):
    other = ToroidalField.from_modes({(1, -3): 0.2j}, nfp=3)
    theta, zeta = np.random.default_rng(5).uniform(0, 2, (2, 23))
    for operation in (operator.add, operator.sub, operator.mul):
        result = operation(field, other)
        assert_allclose(result(theta, zeta), operation(field(theta, zeta), other(theta, zeta)), rtol=1e-12, atol=1e-13)
    assert ((field + other).mpol, (field + other).ntor) == (2, 3)
    assert ((field * other).mpol, (field * other).ntor) == (3, 5)
    for result, expected in [
        (3 + field, 3 + field(theta, zeta)),
        (3 - field, 3 - field(theta, zeta)),
        (3 * field, 3 * field(theta, zeta)),
        (field / 3, field(theta, zeta) / 3),
    ]:
        assert_allclose(result(theta, zeta), expected, rtol=1e-13, atol=1e-13)
    f = ToroidalField.from_modes({(2, 1): 0.5}, nfp=3)
    square = f * f
    assert_allclose(square.coefficient(4, 2), 0.25, rtol=0, atol=1e-14)
    assert_allclose(square.mean, 0.5, rtol=0, atol=1e-14)
    assert_allclose(square(theta, zeta), np.cos(2*theta - 3*zeta)**2, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("power", [0, 1, 2, 3, 4, 5])
def test_integer_powers(field, power):
    result = field ** power
    assert (result.mpol, result.ntor) == (field.mpol * power, field.ntor * power)
    assert_allclose(result(0.71, 0.32), field(0.71, 0.32)**power, rtol=1e-12, atol=1e-12)


def test_resolution_normalization_and_angle_changes(field):
    padded = field.with_resolution(4, 3)
    assert_allclose(padded.with_resolution(2, 2).coefficients, field.coefficients, rtol=0, atol=0)
    reduced = field.with_resolution(0, 1)
    assert_allclose(reduced(0.6, 0.7), 2 + 0.6*np.sin(3*0.7), rtol=1e-13, atol=1e-13)
    assert_allclose(field.normalize().mean, 1, rtol=0, atol=1e-14)
    theta, zeta = np.random.default_rng(4).uniform(0, 2, (2, 11))
    assert_allclose(field.shift(0.3, -0.2)(theta, zeta), field(theta+0.3, zeta-0.2), rtol=1e-13, atol=1e-13)
    assert_allclose(field.reverse_poloidal_angle()(theta, zeta), field(np.pi-theta, zeta), rtol=1e-13, atol=1e-13)


def test_nonlinear_approximations_converge_off_grid():
    f = ToroidalField.from_modes({(0, 0): 2, (1, 1): 0.25}, nfp=3)
    theta, zeta = np.random.default_rng(8).uniform(0, 2, (2, 37))
    coarse = f.reciprocal(mpol=2, ntor=2, sample_shape=(9, 9))
    fine = f.reciprocal(mpol=12, ntor=12, sample_shape=(41, 41))
    expected = 1/f(theta, zeta)
    assert np.max(abs(fine(theta, zeta)-expected)) < np.max(abs(coarse(theta, zeta)-expected)) / 1000
    assert_allclose(fine(theta, zeta), expected, rtol=1e-10, atol=1e-11)
    root = f.power(0.5, mpol=12, ntor=12, sample_shape=(41, 41))
    assert_allclose(root(theta, zeta), np.sqrt(f(theta, zeta)), rtol=1e-10, atol=1e-11)


def test_inputs_and_returned_arrays_are_independent():
    data = np.zeros((3, 3), dtype=complex)
    data[1, 1] = 2
    field = ToroidalField(data)
    data[:] = 5
    exported = field.coefficients
    exported[:] = 6
    result = field + 1
    assert field.mean == 2
    assert result.mean == 3


@pytest.mark.parametrize("data", [[], [1, 2, 1], np.zeros((0, 3)), np.zeros((2, 3)), [[1j]], [[np.nan]], [[np.inf]]])
def test_invalid_spectra(data):
    with pytest.raises(ValueError):
        ToroidalField(data)


@pytest.mark.parametrize("nfp,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_invalid_field_periods(nfp, error):
    with pytest.raises(error):
        ToroidalField.constant(1, nfp=nfp)


@pytest.mark.parametrize("samples,error", [(np.zeros((2, 3)), ValueError), (np.zeros((3, 4)), ValueError), (np.zeros((0, 3)), ValueError), ([[np.nan]], ValueError), ([[1j]], TypeError)])
def test_invalid_samples(samples, error):
    with pytest.raises(error):
        ToroidalField.from_samples(samples)


def test_invalid_modes_and_resolution(field):
    for modes in ({(0, -1): 1}, {(-1, 0): 1}, {(0, 0): 1j}):
        with pytest.raises(ValueError):
            ToroidalField.from_modes(modes)
    with pytest.raises(ValueError):
        ToroidalField.from_modes({(2, 1): 1}, mpol=1)
    with pytest.raises(TypeError):
        ToroidalField.from_modes({(1, 0.5): 1})
    with pytest.raises(IndexError):
        field.coefficient(3, 0)
    with pytest.raises(TypeError):
        field.coefficient(1.5, 0)
    with pytest.raises(ValueError):
        field.with_resolution(-1, 1)
    with pytest.raises(ValueError):
        field.sample(3, 3)
    with pytest.raises(ValueError):
        field.derivative(-1)
    with pytest.raises(TypeError):
        field.derivative(0.5)


def test_invalid_angles(field):
    with pytest.raises(TypeError, match="real"):
        field(1j, 0)
    with pytest.raises(ValueError, match="finite"):
        field(0, np.inf)
    with pytest.raises(ValueError):
        field(np.zeros(2), np.zeros(3))


def test_invalid_arithmetic_and_nonlinear_domains(field):
    for operation in (operator.add, operator.sub, operator.mul):
        with pytest.raises(ValueError, match="nfp"):
            operation(field, ToroidalField.constant(1, nfp=2))
    with pytest.raises(ZeroDivisionError):
        field / 0
    with pytest.raises(ZeroDivisionError):
        ToroidalField.from_modes({(1, 1): 0.5}).normalize()
    with pytest.raises(TypeError):
        field / field
    with pytest.raises(TypeError):
        field ** 0.5
    with pytest.raises(ValueError):
        field ** -1
    with pytest.raises(ValueError):
        field + np.inf
    with pytest.raises(TypeError):
        field * 1j
    for shape in [(6, 7), (3, 3), (9,)]:
        with pytest.raises(ValueError):
            field.reciprocal(mpol=2, ntor=2, sample_shape=shape)
    with pytest.raises(ValueError):
        field.reciprocal(mpol=5, ntor=5, sample_shape=(7, 7))
    with pytest.raises(ValueError):
        ToroidalField.constant(0).reciprocal(mpol=0, ntor=0, sample_shape=(1, 1))
    with pytest.raises(ValueError):
        ToroidalField.constant(-1).power(0.5, mpol=0, ntor=0, sample_shape=(1, 1))
