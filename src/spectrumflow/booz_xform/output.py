"""Read standard booz_xform boozmn NetCDF output."""

from __future__ import annotations

from os import PathLike

import numpy as np
from numpy.typing import NDArray

from .._netcdf import _FourierModes, _NetcdfData, _read_fourier, _surface_index
from ..geometry import BoozerSurface
from ..radial_profile import RadialProfile
from ..toroidalfield import ToroidalField

# File p=-nu is precisely the geometry shift omega=phi-zeta_B.
_FIELDS = {
    "r": ("rmnc_b", "rmns_b", False),
    "z": ("zmnc_b", "zmns_b", True),
    "omega": ("pmnc_b", "pmns_b", True),
    "b": ("bmnc_b", "bmns_b", False),
    "jacobian": ("gmn_b", "gmns_b", False),
}
_PROFILE_GRIDS = {
    "phi_b": "full",
    **dict.fromkeys((
        "iota_b", "buco_b", "bvco_b", "pres_b", "beta_b", "phip_b",
    ), "half"),
}


class BoozXformOutput:
    r"""An in-memory snapshot of a standard boozmn NetCDF file.

    theta,zeta are Boozer angles; physical phi=zeta+omega, where the file
    stores omega=p=-nu and nu=zeta_B-phi. ixn_b already includes nfp.
    lasym__logical__ determines whether asymmetric partners are required.

    ns is the original VMEC full-grid count (file ns_b). Spectral arrays
    contain only selected half-grid surfaces, in jlist order. For a legacy
    one-based jlist entry j, s=(j-1.5)/(ns-1), not (j-1)/(ns-1). No spectral
    surface is on the magnetic axis or exactly at the boundary. profile()
    reads the original radial profiles, rather than the packed surface grid.
    Raw variables retain the on-disk axes; field() uses named dimensions.
    Requires spectrumflow[netcdf], without the booz_xform computation package.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        self._data = _NetcdfData.read(path)
        self.nfp = self._data.integer("nfp_b", 1)
        self.ns = self._data.integer("ns_b", 2)
        self.lasym = self._data.asymmetry()
        if self._data.sizes["radius"] != self.ns:
            raise ValueError("radius dimension does not match ns_b")
        # Some standard files give mode lists the singular mn_mode dimension
        # while spectra use mn_modes. Both axis names are specified by format.
        mode_dim = self._data.dimensions["ixm_b"]
        if mode_dim not in (("mn_mode",), ("mn_modes",)):
            raise ValueError("ixm_b must have dimension mn_mode or mn_modes")
        self._modes = _FourierModes(
            self._data.array("ixm_b", mode_dim),
            self._data.array("ixn_b", mode_dim), self.nfp,
        )
        if self._data.sizes["mn_modes"] != self._modes.m.size:
            raise ValueError("mn_modes dimension does not match the mode lists")
        jlist = self._data.array("jlist", ("comput_surfs",))
        if jlist.size == 0 or np.any(jlist != np.rint(jlist)):
            raise ValueError("jlist must contain nonempty integer surface indices")
        if np.any(jlist < 2) or np.any(jlist > self.ns):
            raise ValueError("jlist entries must lie between 2 and ns_b")
        if np.unique(jlist).size != jlist.size:
            raise ValueError("jlist must not contain duplicate surfaces")
        if self._data.sizes["pack_rad"] != jlist.size:
            raise ValueError("pack_rad dimension does not match jlist")
        self._jlist = jlist.astype(int)

    @property
    def variables(self) -> tuple[str, ...]:
        """Names of all original NetCDF variables, including metadata."""
        return tuple(self._data.variables)

    def __getitem__(self, name: str) -> NDArray:
        """Copy a raw variable in its original file axis order and units."""
        return self._data.variables[name].copy()

    @property
    def jlist(self) -> NDArray[np.int64]:
        """Original one-based VMEC file row labels, shape (surface_count,)."""
        return self._jlist.copy()

    @property
    def surface_count(self) -> int:
        """Number of transformed surfaces, distinct from the original ns."""
        return self._jlist.size

    @property
    def s(self) -> NDArray[np.float64]:
        """Normalized toroidal flux of packed spectra, in original jlist order."""
        return (self._jlist - 1.5) / (self.ns - 1)

    def field(self, name: str, surface_index: int = -1) -> ToroidalField:
        """Read r/z [m], omega [rad], b [T], or jacobian [m/T].

        surface_index selects a packed surface in jlist order, allowing
        negative indices. The Jacobian is for (psi,theta_B,zeta_B), where
        psi is toroidal flux divided by 2*pi [Wb], unlike VMEC's s Jacobian.
        Its supplied sign is retained. All modes are preserved.
        """
        cosine, sine, sine_symmetric = _FIELDS[name]
        index = _surface_index(surface_index, self.surface_count)
        return _read_fourier(
            self._data, self._modes, cosine, sine, ("pack_rad", "mn_modes"), index,
            lasym=self.lasym, sine_symmetric=sine_symmetric,
        )

    def surface(self, surface_index: int = -1) -> BoozerSurface:
        """Return a packed Boozer surface with physical phi=zeta+omega."""
        return BoozerSurface(
            self.field("r", surface_index), self.field("z", surface_index),
            self.field("omega", surface_index),
        )

    def profile(self, name: str, *, unit: str | None = None) -> RadialProfile:
        """Read phi_b (full grid) or iota/buco/bvco/pres/beta/phip_b (half grid).

        Half-grid dummy row 0 is omitted. Coordinates are normalized flux s;
        units and values are retained from the file, with optional unit-label
        override. These profiles cover all original radii, including surfaces
        absent from jlist. For a packed surface k, a half-grid profile's value
        is profile.values[jlist[k]-2].
        """
        grid = _PROFILE_GRIDS[name]
        values = self._data.array(
            name, ("radius",), index=slice(1, None) if grid == "half" else None,
        )
        if grid == "full":
            coordinate = np.linspace(0, 1, self.ns)
        else:
            coordinate = (np.arange(1, self.ns) - 0.5) / (self.ns - 1)
        return RadialProfile(
            coordinate, values, name=name,
            unit=self._data.units[name] if unit is None else unit, grid=grid,
        )
