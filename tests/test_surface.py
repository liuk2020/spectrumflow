"""Analytic torus geometry, angle directions, and lossless surface archives."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from spectrumflow import ToroidalField
from spectrumflow.geometry import BoozerSurface, CylindricalSurface, Surface


def circular_fields(nfp):
    return (
        ToroidalField.from_modes({(0, 0): 3, (1, 0): 0.35}, nfp=nfp),
        ToroidalField.from_modes({(0, 0): 0.4, (1, 0): -0.35j}, nfp=nfp),
    )


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("nfp", [1, 4])
def test_circular_torus_position_basis_metric_and_integrals(sign, nfp):
    surface = CylindricalSurface(*circular_fields(nfp), toroidal_sign=sign)
    theta = np.array([0.13, 0.82, 2.1])[:, None]
    zeta = np.array([0.04, 0.27])[None, :]
    r = 3 + 0.7 * np.cos(theta)
    z = 0.4 + 0.7 * np.sin(theta)
    phi = np.broadcast_to(sign * zeta, (3, 2))
    position = surface.position(theta, zeta)
    assert position.shape == (3, 2, 3)
    assert_allclose(position[..., 0], r * np.cos(phi), rtol=1e-13, atol=1e-13)
    assert_allclose(position[..., 1], r * np.sin(phi), rtol=1e-13, atol=1e-13)
    assert_allclose(position[..., 2], np.broadcast_to(z, (3, 2)), rtol=1e-13, atol=1e-13)
    basis = surface.basis(theta, zeta)
    assert basis.shape == (3, 2, 2, 3)
    assert_allclose(basis[..., 0, 0], -0.7*np.sin(theta)*np.cos(phi), rtol=1e-13, atol=1e-13)
    assert_allclose(basis[..., 1, 1], sign*r*np.cos(phi), rtol=1e-13, atol=1e-13)
    metric = surface.metric_tensor(theta, zeta)
    assert metric.shape == (3, 2, 2, 2)
    assert_allclose(metric[..., 0, 0], 0.7**2, rtol=1e-13, atol=1e-13)
    assert_allclose(metric[..., 0, 1], 0, rtol=0, atol=1e-13)
    assert_allclose(metric[..., 1, 1], np.broadcast_to(r**2, (3, 2)), rtol=1e-13, atol=1e-13)
    expected = (4*np.pi**2*3*0.7, 2*np.pi**2*3*0.7**2)
    assert_allclose(surface.area_volume(17, 13), expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("omega_sign", [-1, 1])
def test_boozer_metric_against_analytic_reparametrized_torus(sign, omega_sign):
    nfp = 3
    omega = omega_sign * ToroidalField.from_modes(
        {(0, 0): 0.12, (1, 0): -0.1j, (1, 1): -0.02j}, nfp=nfp
    )
    surface = BoozerSurface(*circular_fields(nfp), omega, toroidal_sign=sign)
    theta = np.array([0.1, 0.7, 2.2])[:, None]
    zeta = np.array([0.2, 0.5])[None, :]
    phase = theta - nfp*zeta
    r = 3 + 0.7*np.cos(theta)
    phi = sign*zeta + omega_sign*(0.12 + 0.2*np.sin(theta) + 0.04*np.sin(phase))
    phi_theta = omega_sign*(0.2*np.cos(theta) + 0.04*np.cos(phase))
    phi_zeta = sign - omega_sign*0.04*nfp*np.cos(phase)
    assert_allclose(surface.phi(theta, zeta), phi, rtol=1e-13, atol=1e-13)
    expected = (
        0.7**2 + r**2*phi_theta**2,
        r**2*phi_theta*phi_zeta,
        r**2*phi_zeta**2,
    )
    tensor = surface.metric_tensor(theta, zeta)
    for field, values, (a, b) in zip(surface.metric(), expected, ((0, 0), (0, 1), (1, 1))):
        assert_allclose(field(theta, zeta), values, rtol=1e-12, atol=1e-12)
        assert_allclose(tensor[..., a, b], values, rtol=1e-12, atol=1e-12)
    assert_allclose(tensor[..., 1, 0], tensor[..., 0, 1], rtol=0, atol=1e-14)
    assert_allclose(
        surface.area_volume(31, 25),
        (4*np.pi**2*3*0.7, 2*np.pi**2*3*0.7**2), rtol=1e-12, atol=1e-12,
    )


def nonaxisymmetric_surface(kind, sign):
    r = ToroidalField.from_modes({(0, 0): 3, (1, 0): 0.3, (2, -1): 0.02j}, nfp=3)
    z = ToroidalField.from_modes({(1, 0): -0.3j, (1, 2): 0.01 + 0.03j}, nfp=3)
    if kind is CylindricalSurface:
        return kind(r, z, toroidal_sign=sign)
    omega = ToroidalField.from_modes({(0, 0): 0.17, (3, 1): -0.01j}, nfp=3)
    return kind(r, z, omega, toroidal_sign=sign)


@pytest.mark.parametrize("kind", [CylindricalSurface, BoozerSurface])
@pytest.mark.parametrize("sign", [-1, 1])
def test_cross_section_inverts_physical_angle_with_broadcasting(kind, sign):
    surface = nonaxisymmetric_surface(kind, sign)
    theta = np.linspace(0, 2*np.pi, 13)[:, None]
    zeta = np.array([-8.1, -0.7, 0, 0.4, 7.2])[None, :]
    phi = surface.phi(theta, zeta)
    expected = surface.rz(theta, zeta)
    actual = surface.cross_section(theta, phi)
    assert actual[0].shape == (13, 5)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(
        surface.cross_section(0.3, surface.phi(0.3, 0.7)),
        surface.rz(0.3, 0.7), rtol=1e-12, atol=1e-12,
    )
    assert surface.cross_section([], [])[0].shape == (0,)


@pytest.mark.parametrize("sign", [-1, 1])
def test_cross_section_with_large_angle_shift_and_zero_shift(sign):
    source = nonaxisymmetric_surface(CylindricalSurface, sign)
    theta = np.linspace(0, 2*np.pi, 17)
    for shift in (0, 5.7):
        surface = BoozerSurface(
            source.r, source.z, ToroidalField.constant(shift, nfp=source.nfp),
            toroidal_sign=sign,
        )
        assert_allclose(
            surface.cross_section(theta, 0.2),
            source.rz(theta, sign*(0.2-shift)), rtol=1e-12, atol=1e-12,
        )


def test_cross_section_rejects_invalid_physical_angles():
    surface = nonaxisymmetric_surface(BoozerSurface, 1)
    with pytest.raises(TypeError, match="real"):
        surface.cross_section(0, 1j)
    with pytest.raises(ValueError, match="finite"):
        surface.cross_section(0, np.nan)
    with pytest.raises(ValueError):
        surface.cross_section(np.zeros(2), np.zeros(3))


@pytest.mark.parametrize("kind", [CylindricalSurface, BoozerSurface])
@pytest.mark.parametrize("sign", [-1, 1])
def test_nonaxisymmetric_basis_matches_position_derivatives_and_periodicity(kind, sign):
    surface = nonaxisymmetric_surface(kind, sign)
    theta, zeta = np.random.default_rng(13).uniform(0.2, 1.5, (2, 11))
    h = 1e-6
    expected = np.stack((
        (surface.position(theta+h, zeta) - surface.position(theta-h, zeta))/(2*h),
        (surface.position(theta, zeta+h) - surface.position(theta, zeta-h))/(2*h),
    ), axis=-2)
    assert_allclose(surface.basis(theta, zeta), expected, rtol=2e-8, atol=2e-9)
    tensor = np.einsum("...ai,...bi->...ab", expected, expected)
    for field, (a, b) in zip(surface.metric(), ((0, 0), (0, 1), (1, 1))):
        assert_allclose(field(theta, zeta), tensor[..., a, b], rtol=5e-8, atol=1e-8)
        assert_allclose(field(theta+2*np.pi, zeta+2*np.pi/3), field(theta, zeta), rtol=1e-12, atol=1e-12)
    period = 2*np.pi/surface.nfp
    xyz = surface.position(theta, zeta)
    angle = sign*period
    rotated = np.stack((
        xyz[..., 0]*np.cos(angle) - xyz[..., 1]*np.sin(angle),
        xyz[..., 0]*np.sin(angle) + xyz[..., 1]*np.cos(angle), xyz[..., 2],
    ), axis=-1)
    assert_allclose(surface.position(theta, zeta+period), rotated, rtol=1e-12, atol=1e-12)
    assert_allclose(surface.position(theta, zeta+2*np.pi), xyz, rtol=1e-12, atol=1e-12)


def test_broadcasting_zero_shift_and_explicit_resolution():
    r, z = circular_fields(2)
    cylinder = CylindricalSurface(r, z)
    boozer = BoozerSurface(r, z, ToroidalField.constant(0, nfp=2, mpol=3, ntor=2))
    assert (boozer.mpol, boozer.ntor) == (3, 2)
    theta, zeta = np.zeros((2, 1, 1)), np.zeros((1, 3, 4))
    for method in ("position", "basis", "metric_tensor"):
        assert_allclose(getattr(cylinder, method)(theta, zeta), getattr(boozer, method)(theta, zeta), rtol=0, atol=1e-14)
    assert cylinder.position(0, 0).shape == (3,)
    assert cylinder.basis(0, 0).shape == (2, 3)
    assert cylinder.metric_tensor(0, 0).shape == (2, 2)
    assert cylinder.position([], []).shape == (0, 3)
    assert cylinder.phi(theta, zeta).shape == (2, 3, 4)
    padded = boozer.with_resolution(4, 3)
    assert type(padded) is BoozerSurface
    assert (padded.omega.mpol, padded.omega.ntor) == (4, 3)
    assert_allclose(padded.position(0.3, 0.2), boozer.position(0.3, 0.2), rtol=1e-13, atol=1e-13)
    reduced = boozer.with_resolution(0, 0)
    assert reduced.r.mean == 3
    assert (boozer.r.mpol, boozer.omega.mpol) == (1, 3)


@pytest.mark.parametrize("kind", [CylindricalSurface, BoozerSurface])
@pytest.mark.parametrize("sign", [-1, 1])
def test_npz_round_trip_preserves_all_modes_and_conventions(tmp_path, kind, sign):
    original = nonaxisymmetric_surface(kind, sign)
    path = tmp_path / "surface.data"
    original.save(path)
    assert path.is_file()
    assert not (tmp_path / "surface.data.npz").exists()
    restored = Surface.load(path)
    assert type(restored) is kind
    assert restored.nfp == 3
    assert restored.toroidal_sign == sign
    names = ("r", "z", "omega") if kind is BoozerSurface else ("r", "z")
    for name in names:
        assert_array_equal(getattr(restored, name).coefficients, getattr(original, name).coefficients)
    assert_allclose(restored.position(0.41, 0.23), original.position(0.41, 0.23), rtol=0, atol=0)
    assert type(kind.load(path)) is kind
    other = CylindricalSurface if kind is BoozerSurface else BoozerSurface
    with pytest.raises(ValueError, match="archive does not contain"):
        other.load(path)
    with np.load(path, allow_pickle=False) as data:
        assert data["phase"].item() == "exp(i*(m*theta-n*nfp*zeta))"
        assert data["length_unit"].item() == "m"
        assert data["angle_unit"].item() == "rad"


@pytest.mark.parametrize("key,value", [
    ("version", 2), ("version", 1.0), ("coordinates", "unknown"),
    ("phase", "opposite sign"), ("length_unit", "cm"), ("angle_unit", "deg"),
    ("nfp", 0), ("toroidal_sign", 0), ("toroidal_sign", True),
    ("nfp", [3]), ("r_coefficients", [[1j]]),
])
def test_npz_rejects_invalid_metadata_and_coefficients(tmp_path, key, value):
    path = tmp_path / "surface.npz"
    nonaxisymmetric_surface(BoozerSurface, -1).save(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays[key] = np.asarray(value)
    np.savez(path, **arrays)
    with pytest.raises((ValueError, TypeError)):
        Surface.load(path)


def test_npz_rejects_missing_fields(tmp_path):
    path = tmp_path / "surface.npz"
    nonaxisymmetric_surface(BoozerSurface, 1).save(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    del arrays["omega_coefficients"]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="missing"):
        Surface.load(path)


def test_incompatible_fields_and_invalid_inputs():
    r, z = circular_fields(3)
    with pytest.raises(ValueError, match="nfp"):
        CylindricalSurface(r, ToroidalField.constant(0, nfp=2))
    with pytest.raises(ValueError, match="nfp"):
        BoozerSurface(r, z, ToroidalField.constant(0))
    with pytest.raises(TypeError, match="ToroidalField"):
        CylindricalSurface(r, np.zeros(3))
    for sign, error in [(0, ValueError), (2, ValueError), (True, TypeError), (1.0, TypeError)]:
        with pytest.raises(error, match="toroidal_sign"):
            CylindricalSurface(r, z, toroidal_sign=sign)
    surface = CylindricalSurface(r, z)
    with pytest.raises(ValueError, match="resolve"):
        surface.area_volume(2, 3)
    with pytest.raises(ValueError, match="resolve"):
        surface.area_volume(3, 0)
    with pytest.raises(TypeError, match="integer"):
        surface.area_volume(3.5, 3)
    with pytest.raises(TypeError, match="real"):
        surface.phi(1j, 0)
    with pytest.raises(ValueError, match="finite"):
        surface.phi(0, np.inf)
    with pytest.raises(ValueError):
        surface.position(np.zeros(2), np.zeros(3))
