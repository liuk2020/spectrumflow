"""Physical toroidal cuts, closed R-Z curves and local plotting styles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrumflow import ToroidalField
from spectrumflow.geometry import BoozerSurface, CylindricalSurface


def make_surface(kind, sign):
    r = ToroidalField.from_modes({(0, 0): 3, (1, 0): 0.35, (0, 1): 0.1j}, nfp=3)
    z = ToroidalField.from_modes({(1, 0): -0.35j, (1, 1): 0.05}, nfp=3)
    if kind is CylindricalSurface:
        return kind(r, z, toroidal_sign=sign)
    omega = ToroidalField.from_modes({(0, 0): 0.2, (1, 0): -0.1j}, nfp=3)
    return kind(r, z, omega, toroidal_sign=sign)


@pytest.mark.parametrize("kind", [CylindricalSurface, BoozerSurface])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("phi", [None, 0.31, [-0.2, 0.4, 7.1]])
def test_sections_use_physical_phi_and_close_curves(kind, sign, phi):
    surface = make_surface(kind, sign)
    fig, ax = plt.subplots()
    other = plt.figure()
    original_style = dict(matplotlib.rcParams)
    returned, lines = surface.plot_cross_section(
        phi, ax=ax, ntheta=33, usetex=False, linewidth=2,
    )
    cuts = [0, np.pi/3] if phi is None else np.atleast_1d(phi)
    theta = np.linspace(0, 2*np.pi, 33)
    assert returned is ax and len(lines) == len(cuts)
    assert not other.axes
    for angle, line in zip(cuts, lines):
        shift = 0 if kind is CylindricalSurface else 0.2 + 0.2*np.sin(theta)
        zeta = sign*(angle-shift)
        expected_r = 3 + 0.7*np.cos(theta) + 0.2*np.sin(3*zeta)
        expected_z = 0.7*np.sin(theta) + 0.1*np.cos(theta-3*zeta)
        assert_allclose(line.get_xdata(), expected_r, rtol=1e-12, atol=1e-12)
        assert_allclose(line.get_ydata(), expected_z, rtol=1e-12, atol=1e-12)
        assert_allclose(line.get_xydata()[0], line.get_xydata()[-1], rtol=0, atol=1e-12)
        assert line.axes is ax and line.get_linewidth() == 2
    assert ax.get_aspect() == 1
    assert ax.get_xlabel() == r"$R$" and ax.get_ylabel() == r"$Z$"
    assert ax.get_title() == ""
    fig.canvas.draw()
    for axis in (ax.xaxis, ax.yaxis):
        assert axis.label.get_fontsize() == 17
        assert all(label.get_fontsize() == 17 for label in axis.get_majorticklabels())
        assert not axis.label.get_usetex()
    assert dict(matplotlib.rcParams) == original_style
    plt.close(fig)
    plt.close(other)


def test_default_tex_persists_and_style_is_adjustable():
    surface = make_surface(CylindricalSurface, 1)
    original_style = dict(matplotlib.rcParams)
    ax, _ = surface.plot_cross_section(figsize=(7, 5))
    for axis in (ax.xaxis, ax.yaxis):
        assert axis.label.get_fontsize() == 17
        assert axis.label.get_usetex()
        assert axis.get_offset_text().get_usetex()
        assert all(label.get_usetex() for label in axis.get_majorticklabels())
        assert all(label.get_text().startswith("$") for label in axis.get_majorticklabels())
    assert_allclose(ax.figure.get_size_inches(), [7, 5], rtol=0, atol=1e-14)
    surface.plot_cross_section(0, ax=ax, fontsize=20, tick_fontsize=13, usetex=False)
    ax.figure.canvas.draw()
    assert ax.xaxis.label.get_fontsize() == 20
    assert all(label.get_fontsize() == 13 for label in ax.get_xticklabels())
    assert dict(matplotlib.rcParams) == original_style
    plt.close(ax.figure)


@pytest.mark.parametrize("phi,error", [
    ([], ValueError), ([[0, 1]], ValueError), (np.nan, ValueError), (1j, TypeError),
])
def test_invalid_section_angles(phi, error):
    with pytest.raises(error):
        make_surface(CylindricalSurface, 1).plot_cross_section(phi)


@pytest.mark.parametrize("ntheta,error", [(2, ValueError), (3.5, TypeError), (True, TypeError)])
def test_invalid_sampling(ntheta, error):
    with pytest.raises(error, match="ntheta"):
        make_surface(CylindricalSurface, 1).plot_cross_section(ntheta=ntheta)
