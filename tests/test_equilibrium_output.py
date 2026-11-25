"""Synthetic wout/boozmn files with explicit signs, mode lists and radial grids."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from spectrumflow.VMEC import VmecOutput
from spectrumflow.booz_xform import BoozXformOutput
from spectrumflow.geometry import BoozerSurface, CylindricalSurface

netcdf = pytest.importorskip("netCDF4")

NFP, NS = 3, 5
M = np.array([2, 0, 1, 0, 1])
XN = np.array([6, 3, -3, 0, 0])
M_NYQ = np.array([3, 0, 0, 1, 2, 1])
XN_NYQ = np.array([-6, 0, -3, 3, 0, 0])
VMEC_FIELDS = {
    "r": ("rmnc", "rmns", "full", False, False),
    "z": ("zmnc", "zmns", "full", False, True),
    "lambda": ("lmnc", "lmns", "half", False, True),
    "b": ("bmnc", "bmns", "half", True, False),
    "jacobian": ("gmnc", "gmns", "half", True, False),
    "bsup_theta": ("bsupumnc", "bsupumns", "half", True, False),
    "bsup_zeta": ("bsupvmnc", "bsupvmns", "half", True, False),
    "bsub_theta": ("bsubumnc", "bsubumns", "half", True, False),
    "bsub_zeta": ("bsubvmnc", "bsubvmns", "half", True, False),
    "bsub_s": ("bsubsmnc", "bsubsmns", "full", True, True),
}
BOOZ_FIELDS = {
    "r": ("rmnc_b", "rmns_b", False),
    "z": ("zmnc_b", "zmns_b", True),
    "omega": ("pmnc_b", "pmns_b", True),
    "b": ("bmnc_b", "bmns_b", False),
    "jacobian": ("gmn_b", "gmns_b", False),
}


def put(dataset, name, values, dims=(), unit=""):
    values = np.asarray(values)
    var = dataset.createVariable(name, "i4" if values.dtype.kind == "i" else "f8", dims)
    var[...] = values
    if unit:
        var.units = unit


def spectra(m, xn, nr):
    cosine = np.arange(1, m.size + 1, dtype=float) * 0.02
    sine = np.arange(1, m.size + 1, dtype=float) * -0.03
    zero = (m == 0) & (xn == 0)
    cosine[zero], sine[zero] = 3, 0
    factor = 1 + np.arange(nr)[:, None] * 0.1
    return factor * cosine, factor * sine


def write_wout(path, *, lasym=False, transpose=False, format="NETCDF3_CLASSIC"):
    expected = {}
    with netcdf.Dataset(path, "w", format=format) as d:
        for dim, size in (("radius", NS), ("mn_mode", M.size), ("mn_mode_nyq", M_NYQ.size)):
            d.createDimension(dim, size)
        for name, value in (("nfp", NFP), ("ns", NS), ("lasym__logical__", int(lasym))):
            put(d, name, value)
        for name, values, dim in (
            ("xm", M, "mn_mode"), ("xn", XN, "mn_mode"),
            ("xm_nyq", M_NYQ, "mn_mode_nyq"), ("xn_nyq", XN_NYQ, "mn_mode_nyq"),
        ):
            put(d, name, values, (dim,))
        for name, (cos_name, sin_name, _, nyq, sine_symmetric) in VMEC_FIELDS.items():
            m, xn = (M_NYQ, XN_NYQ) if nyq else (M, XN)
            cosine, sine = spectra(m, xn, NS)
            if not lasym:
                if sine_symmetric:
                    cosine[:] = 0
                else:
                    sine[:] = 0
            expected[name] = (cosine, sine, m, xn)
            dim = "mn_mode_nyq" if nyq else "mn_mode"
            dims = (dim, "radius") if transpose else ("radius", dim)
            if lasym or not sine_symmetric:
                put(d, cos_name, cosine.T if transpose else cosine, dims)
            if lasym or sine_symmetric:
                put(d, sin_name, sine.T if transpose else sine, dims)
        put(d, "presf", [10, 8, 6, 4, 2], ("radius",), "Pa")
        put(d, "iotas", [999, 0.2, 0.3, 0.4, 0.5], ("radius",))
        put(d, "custom", [999, 2, 4, 6, 8], ("radius",), "m")
    return expected


def write_boozmn(
    path, *, lasym=False, transpose=False, mode_dim="mn_mode", jlist=(2, 4, 5),
    format="NETCDF3_CLASSIC",
):
    expected = {}
    with netcdf.Dataset(path, "w", format=format) as d:
        for dim, size in (
            ("radius", NS), ("mn_modes", M.size),
            ("pack_rad", len(jlist)), ("comput_surfs", len(jlist)),
        ):
            d.createDimension(dim, size)
        if mode_dim != "mn_modes":
            d.createDimension(mode_dim, M.size)
        for name, value in (("nfp_b", NFP), ("ns_b", NS), ("lasym__logical__", int(lasym))):
            put(d, name, value)
        put(d, "ixm_b", M, (mode_dim,))
        put(d, "ixn_b", XN, (mode_dim,))
        put(d, "jlist", np.array(jlist, dtype=int), ("comput_surfs",))
        for name, (cos_name, sin_name, sine_symmetric) in BOOZ_FIELDS.items():
            cosine, sine = spectra(M, XN, len(jlist))
            if not lasym:
                if sine_symmetric:
                    cosine[:] = 0
                else:
                    sine[:] = 0
            expected[name] = (cosine, sine, M, XN)
            dims = ("mn_modes", "pack_rad") if transpose else ("pack_rad", "mn_modes")
            if lasym or not sine_symmetric:
                put(d, cos_name, cosine.T if transpose else cosine, dims)
            if lasym or sine_symmetric:
                put(d, sin_name, sine.T if transpose else sine, dims)
        put(d, "phi_b", np.linspace(0, -2, NS), ("radius",), "Wb")
        put(d, "iota_b", [999, 0.2, 0.3, 0.4, 0.5], ("radius",))
        put(d, "buco_b", [999, 1, 2, 3, 4], ("radius",), "T m")
    return expected


def analytic_series(data, index, theta, zeta):
    cosine, sine, m, xn = data
    phase = m[:, None, None] * theta - xn[:, None, None] * zeta
    return np.sum(
        cosine[index, :, None, None] * np.cos(phase)
        + sine[index, :, None, None] * np.sin(phase), axis=0,
    )


def check_fields(output, expected):
    theta = np.array([0.13, 0.72, 1.94])[:, None]
    zeta = np.array([0.04, 0.27])[None, :]
    for name, data in expected.items():
        for index in (1, -1):
            field = output.field(name, index)
            assert field.nfp == NFP
            assert field.mpol == max(data[2])
            assert field.ntor == max(abs(data[3])) // NFP
            assert_allclose(
                field(theta, zeta), analytic_series(data, index, theta, zeta),
                rtol=1e-13, atol=1e-13,
            )
            assert_allclose(
                field(theta, zeta + 2 * np.pi / NFP), field(theta, zeta),
                rtol=1e-13, atol=1e-13,
            )
            # Check the nfp factor and sign with an independent analytic derivative.
            cosine, sine, m, xn = data
            phase = m[:, None, None] * theta - xn[:, None, None] * zeta
            dzeta = np.sum(xn[:, None, None] * (
                cosine[index, :, None, None] * np.sin(phase)
                - sine[index, :, None, None] * np.cos(phase)
            ), axis=0)
            assert_allclose(field.derivative(zeta_order=1)(theta, zeta), dzeta,
                            rtol=1e-13, atol=1e-13)
            zero = (m == 0) & (xn == 0)
            assert_allclose(field.mean, cosine[index, zero].item(), rtol=0, atol=0)


@pytest.mark.parametrize("lasym", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("format", ["NETCDF3_CLASSIC", "NETCDF4"])
def test_vmec_native_grids_and_unordered_mode_conversion(tmp_path, lasym, transpose, format):
    path = tmp_path / "wout.nc"
    expected = write_wout(path, lasym=lasym, transpose=transpose, format=format)
    output = VmecOutput(path)
    path.unlink()  # Reading eagerly closes the file and owns the data.
    assert output.nfp == NFP and output.ns == NS and output.lasym == lasym
    check_fields(output, expected)
    surface = output.surface()
    assert isinstance(surface, CylindricalSurface)
    assert surface.phi(0.2, 0.4) == 0.4
    assert_allclose(surface.r(0.2, 0.4), output.field("r")(0.2, 0.4), rtol=0, atol=0)
    assert_allclose(output.s_full, [0, 0.25, 0.5, 0.75, 1], rtol=0, atol=0)
    assert_allclose(output.s_half, [0.125, 0.375, 0.625, 0.875], rtol=0, atol=0)
    for name in ("lambda", "b", "jacobian", "bsup_theta", "bsub_theta"):
        with pytest.raises(ValueError, match="half-grid"):
            output.field(name, 0)
    assert_allclose(output.field("bsub_s", 0)(0.2, 0.4),
                    analytic_series(expected["bsub_s"], 0, 0.2, 0.4).item(),
                    rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("lasym", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("mode_dim", ["mn_mode", "mn_modes"])
def test_booz_sparse_surfaces_and_physical_angle_shift(tmp_path, lasym, transpose, mode_dim):
    path = tmp_path / "boozmn.nc"
    expected = write_boozmn(path, lasym=lasym, transpose=transpose, mode_dim=mode_dim)
    output = BoozXformOutput(path)
    path.unlink()
    assert output.nfp == NFP and output.ns == NS and output.surface_count == 3
    assert output.lasym == lasym
    assert_array_equal(output.jlist, [2, 4, 5])
    assert_allclose(output.s, [0.125, 0.625, 0.875], rtol=0, atol=0)
    check_fields(output, expected)
    surface = output.surface(1)
    assert isinstance(surface, BoozerSurface)
    theta, zeta = np.array([0.1, 0.6])[:, None], np.array([0.03, 0.15])[None, :]
    # pmns_b=-numns_b=phi-zeta_B; reversing it would produce the wrong positions.
    phi = zeta + analytic_series(expected["omega"], 1, theta, zeta)
    assert_allclose(surface.phi(theta, zeta), phi, rtol=1e-13, atol=1e-13)
    r = analytic_series(expected["r"], 1, theta, zeta)
    z = analytic_series(expected["z"], 1, theta, zeta)
    xyz = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)
    assert_allclose(surface.position(theta, zeta), xyz, rtol=1e-13, atol=1e-13)


def test_native_profiles_units_and_copy_ownership(tmp_path):
    vmec_path, booz_path = tmp_path / "wout.nc", tmp_path / "boozmn.nc"
    write_wout(vmec_path)
    write_boozmn(booz_path, format="NETCDF4")
    vmec, booz = VmecOutput(vmec_path), BoozXformOutput(booz_path)
    pressure = vmec.profile("presf")
    assert pressure.grid == "full" and pressure.unit == "Pa"
    assert_array_equal(pressure.coordinate, vmec.s_full)
    assert_array_equal(pressure.values, [10, 8, 6, 4, 2])
    iota = vmec.profile("iotas")
    assert iota.grid == "half"
    assert_array_equal(iota.coordinate, vmec.s_half)
    assert_allclose(iota.values, [0.2, 0.3, 0.4, 0.5], rtol=0, atol=0)
    with pytest.raises(ValueError, match="specify grid"):
        vmec.profile("custom")
    assert_array_equal(vmec.profile("custom", grid="half").values, [2, 4, 6, 8])
    with pytest.raises(ValueError, match="full grid"):
        vmec.profile("presf", grid="half")
    assert booz.profile("buco_b").unit == "T m"
    assert vmec.profile("custom", grid="full", unit="cm").unit == "cm"
    booz_iota = booz.profile("iota_b")
    assert_array_equal(booz_iota.coordinate, vmec.s_half)
    assert_allclose(booz_iota(booz.s), [0.2, 0.4, 0.5], rtol=1e-14, atol=1e-14)
    flux = booz.profile("phi_b")
    assert flux.grid == "full" and flux.unit == "Wb"
    assert_array_equal(flux.coordinate, vmec.s_full)
    assert_array_equal(flux.values, [0, -0.5, -1, -1.5, -2])
    for output, raw_name in ((vmec, "rmnc"), (booz, "rmnc_b")):
        original = output[raw_name]
        output[raw_name][:] = 0
        assert_array_equal(output[raw_name], original)
        assert raw_name in output.variables
        with pytest.raises(IndexError):
            output.surface(999)
        with pytest.raises(TypeError, match="integer"):
            output.surface(0.5)
    booz.jlist[:] = 2
    assert_array_equal(booz.jlist, [2, 4, 5])


def test_unused_half_grid_rows_can_contain_missing_data(tmp_path):
    vmec_path, booz_path = tmp_path / "wout.nc", tmp_path / "boozmn.nc"
    write_wout(vmec_path)
    write_boozmn(booz_path)
    with netcdf.Dataset(vmec_path, "a") as d:
        d["iotas"][0] = np.nan
        d["bmnc"][0, :] = np.ma.masked
    with netcdf.Dataset(booz_path, "a") as d:
        d["iota_b"][0] = np.ma.masked
    vmec, booz = VmecOutput(vmec_path), BoozXformOutput(booz_path)
    assert_array_equal(vmec.profile("iotas").values, booz.profile("iota_b").values)
    assert np.isfinite(vmec.field("b", 1)(0.2, 0.4))


@pytest.mark.parametrize("reader,writer", [
    (VmecOutput, write_wout), (BoozXformOutput, write_boozmn),
])
def test_named_axes_and_missing_values_are_validated(tmp_path, reader, writer):
    path = tmp_path / "output.nc"
    writer(path)
    raw_name = "rmnc" if reader is VmecOutput else "rmnc_b"
    with netcdf.Dataset(path, "a") as d:
        d[raw_name][-1, 0] = np.ma.masked
    with pytest.raises(ValueError, match="missing data"):
        reader(path).field("r")
    writer(path)
    with netcdf.Dataset(path, "a") as d:
        dims, values = d[raw_name].dimensions, d[raw_name][...]
        # Wrong axis names with identical lengths must not be guessed by shape.
        d.createDimension("unknown_radius", values.shape[0])
        d.renameVariable(raw_name, "old_coefficients")
        put(d, raw_name, values, ("unknown_radius", dims[1]))
    with pytest.raises(ValueError, match="dimensions"):
        reader(path).field("r")


@pytest.mark.parametrize("name,value,match", [
    ("nfp", 0, "nfp"), ("ns", 1, "ns"), ("ns", 6, "radius"),
    ("lasym__logical__", 2, "logical"),
    ("xm", [2.5, 0, 1, 0, 1], "integers"),
    ("xn", [6, 2, -3, 0, 0], "multiple"),
    ("xn", [6, 3, 0, 0, 0], "duplicate"),
])
def test_vmec_rejects_invalid_metadata(tmp_path, name, value, match):
    path = tmp_path / "wout.nc"
    write_wout(path)
    with netcdf.Dataset(path, "a") as d:
        if name == "xm":
            # Mode variables can be floating point in real wout files.
            d.renameVariable("xm", "old_xm")
            put(d, "xm", np.array(value), ("mn_mode",))
        else:
            d[name][...] = value
    with pytest.raises(ValueError, match=match):
        VmecOutput(path)


@pytest.mark.parametrize("jlist", [(1, 4, 5), (2, 4, 6), (2, 2, 5)])
def test_booz_rejects_invalid_surface_indices(tmp_path, jlist):
    path = tmp_path / "boozmn.nc"
    write_boozmn(path, jlist=jlist)
    with pytest.raises(ValueError, match="jlist"):
        BoozXformOutput(path)


@pytest.mark.parametrize("reader,writer,cos_name,sin_name", [
    (VmecOutput, write_wout, "rmnc", "rmns"),
    (BoozXformOutput, write_boozmn, "rmnc_b", "rmns_b"),
])
def test_missing_asymmetric_partners_and_invalid_spectra_are_not_silenced(
    tmp_path, reader, writer, cos_name, sin_name,
):
    path = tmp_path / "output.nc"
    writer(path, lasym=True)
    with netcdf.Dataset(path, "a") as d:
        d.renameVariable(sin_name, "removed_partner")
    with pytest.raises(KeyError, match=sin_name):
        reader(path).field("r")
    writer(path)
    with netcdf.Dataset(path, "a") as d:
        d[cos_name][-1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reader(path).field("r")
    writer(path, lasym=True)
    with netcdf.Dataset(path, "a") as d:
        d["lasym__logical__"][...] = 0
    with pytest.raises(ValueError, match="nonzero"):
        reader(path).field("r")
    writer(path, lasym=True)
    with netcdf.Dataset(path, "a") as d:
        d[sin_name][-1, 3] = 1  # Zero mode is deliberately not the first entry.
    with pytest.raises(ValueError, match="zero-mode sine"):
        reader(path).field("r")
