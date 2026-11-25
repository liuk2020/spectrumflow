"""Explicit radial coordinates, scalar interpolation and adjustable plotting."""

import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from spectrumflow import RadialProfile


def test_profile_owns_arrays_and_interpolates_with_input_shape():
    coordinate = np.array([0.1, 0.3, 0.8])
    values = 2 + 3 * coordinate
    profile = RadialProfile(coordinate, values, name="pressure", unit="Pa", grid="half")
    coordinate[:] = 0
    values[:] = 0
    profile.coordinate[:] = 0
    profile.values[:] = 0
    assert_array_equal(profile.coordinate, [0.1, 0.3, 0.8])
    points = np.array([[0.1, 0.2], [0.5, 0.8]])
    assert_allclose(profile(points), 2 + 3 * points, rtol=1e-14, atol=1e-14)
    assert_allclose(profile(0.4), 3.2, rtol=1e-14, atol=1e-14)
    assert profile(0.4).shape == ()
    assert profile([]).shape == (0,)
    assert (profile.name, profile.unit, profile.grid) == ("pressure", "Pa", "half")


@pytest.mark.parametrize("coordinate,values,error,match", [
    ([], [], ValueError, "nonempty"),
    ([[0, 1]], [[1, 2]], ValueError, "1D"),
    ([0, 1], [1], ValueError, "shape"),
    ([0, 0], [1, 2], ValueError, "increasing"),
    ([1, 0], [1, 2], ValueError, "increasing"),
    ([0, np.nan], [1, 2], ValueError, "finite"),
    ([0, 1], [np.inf, 2], ValueError, "finite"),
    ([0, 1j], [1, 2], TypeError, "real"),
    ([0, 1], [1, 2j], TypeError, "real"),
    (np.ma.array([0, 1], mask=[0, 1]), [1, 2], ValueError, "missing"),
    ([0, 1], np.ma.array([1, 2], mask=[0, 1]), ValueError, "missing"),
])
def test_invalid_profiles(coordinate, values, error, match):
    with pytest.raises(error, match=match):
        RadialProfile(coordinate, values)


@pytest.mark.parametrize("point,error,match", [
    (-1e-6, ValueError, "outside"),
    (1.000001, ValueError, "outside"),
    (np.nan, ValueError, "finite"),
    (1j, TypeError, "real"),
    (np.ma.array([0.5], mask=[1]), ValueError, "missing"),
])
def test_evaluation_rejects_extrapolation_and_invalid_coordinates(point, error, match):
    with pytest.raises(error, match=match):
        RadialProfile([0, 1], [2, 3])(point)


def test_one_point_profile_has_no_implicit_extrapolation():
    profile = RadialProfile([0.4], [7])
    assert_allclose(profile([0.4, 0.4]), [7, 7], rtol=0, atol=0)
    with pytest.raises(ValueError, match="outside"):
        profile(0.5)


def test_plot_returns_line_with_units_and_keeps_global_style():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = dict(matplotlib.rcParams)
    profile = RadialProfile(
        [0, 0.2, 0.7], [8, 6, 3], name="pressure", unit="Pa",
        coordinate_name="r", coordinate_unit="m",
    )
    fig, ax = plt.subplots()
    returned_ax, line = profile.plot(ax=ax, color="red", linestyle="--")
    assert returned_ax is ax
    assert line is ax.lines[0]
    assert line.get_color() == "red"
    assert ax.get_xlabel() == "r [m]"
    assert ax.get_ylabel() == "pressure [Pa]"
    assert_array_equal(line.get_xdata(), profile.coordinate)
    assert_array_equal(line.get_ydata(), profile.values)
    profile.plot(ax=ax, xlabel="flux", ylabel="p")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("flux", "p")
    fig.canvas.draw()
    assert dict(matplotlib.rcParams) == before
    plt.close(fig)


def test_core_and_reader_imports_do_not_require_optional_io_or_plotting():
    # A fresh interpreter verifies actual import isolation, not just cached imports.
    result = subprocess.run(
        [sys.executable, "-c", """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'netCDF4', 'xarray', 'booz_xform', 'matplotlib'}:
        raise ImportError('blocked optional dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from spectrumflow import RadialProfile, ToroidalField
from spectrumflow.VMEC import VmecOutput
from spectrumflow.booz_xform import BoozXformOutput
assert RadialProfile([0, 1], [2, 3])(0.5) == 2.5
for reader in (VmecOutput, BoozXformOutput):
    try:
        reader('unused.nc')
    except ImportError as error:
        assert 'spectrumflow[netcdf]' in str(error)
    else:
        raise AssertionError('reader should require the netcdf extra')
"""],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
