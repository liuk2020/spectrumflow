"""Read VMEC wout NetCDF files without changing their coordinate directions."""

from __future__ import annotations

from os import PathLike
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._netcdf import _FourierModes, _NetcdfData, _read_fourier, _surface_index
from ..geometry import CylindricalSurface
from ..radial_profile import RadialProfile
from ..toroidalfield import ToroidalField

# cosine, sine, radial grid, mode dimension, symmetric sine series
_FIELDS = {
    "r": ("rmnc", "rmns", "full", "mn_mode", False),
    "z": ("zmnc", "zmns", "full", "mn_mode", True),
    "lambda": ("lmnc", "lmns", "half", "mn_mode", True),
    "b": ("bmnc", "bmns", "half", "mn_mode_nyq", False),
    "jacobian": ("gmnc", "gmns", "half", "mn_mode_nyq", False),
    "bsup_theta": ("bsupumnc", "bsupumns", "half", "mn_mode_nyq", False),
    "bsup_zeta": ("bsupvmnc", "bsupvmns", "half", "mn_mode_nyq", False),
    "bsub_theta": ("bsubumnc", "bsubumns", "half", "mn_mode_nyq", False),
    "bsub_zeta": ("bsubvmnc", "bsubvmns", "half", "mn_mode_nyq", False),
    "bsub_s": ("bsubsmnc", "bsubsmns", "full", "mn_mode_nyq", True),
}
_PROFILE_GRIDS = {
    **dict.fromkeys((
        "iotaf", "q_factor", "presf", "phi", "phipf", "chi", "chipf",
        "jcuru", "jcurv", "jdotb", "bdotb", "bdotgradv",
    ), "full"),
    **dict.fromkeys((
        "iotas", "mass", "pres", "beta_vol", "buco", "bvco", "vp", "specw",
        "phips", "over_r",
    ), "half"),
}


class VmecOutput:
    r"""An in-memory snapshot of a VMEC wout file; requires the netcdf extra.

    theta is VMEC's poloidal parameter, zeta is the physical cylindrical
    angle (phi=zeta); theta*=theta+lambda is the straight-field-line angle.
    File modes use m*theta-xn*zeta, with xn already including nfp. Conversion
    to ToroidalField follows its documented coefficient normalization.
    Stellarator symmetry comes from lasym__logical__, never a user default.

    Full-grid positions are s=j/(ns-1). Half-grid file rows j=1,...,ns-1
    lie at s=(j-1/2)/(ns-1); row 0 is unused. field() uses file row indices,
    including Python negative indices. profile() omits unused half-grid rows.
    No interpolation, angle reversal or Jacobian sign change is implicit.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        self._data = _NetcdfData.read(path)
        self.nfp = self._data.integer("nfp", 1)
        self.ns = self._data.integer("ns", 2)
        self.lasym = self._data.asymmetry()
        if self._data.sizes["radius"] != self.ns:
            raise ValueError("radius dimension does not match ns")
        self._modes = {
            dim: _FourierModes(
                self._data.array(m, (dim,)), self._data.array(n, (dim,)), self.nfp
            )
            for dim, m, n in (
                ("mn_mode", "xm", "xn"),
                ("mn_mode_nyq", "xm_nyq", "xn_nyq"),
            )
        }

    @property
    def variables(self) -> tuple[str, ...]:
        """Names of all original NetCDF variables, including metadata."""
        return tuple(self._data.variables)

    def __getitem__(self, name: str) -> NDArray:
        """Copy a raw variable in its original file axis order and units."""
        return self._data.variables[name].copy()

    @property
    def s_full(self) -> NDArray[np.float64]:
        """Normalized toroidal flux at full-grid rows, shape (ns,)."""
        return np.linspace(0, 1, self.ns)

    @property
    def s_half(self) -> NDArray[np.float64]:
        """Physical half-grid positions without the dummy row, shape (ns-1,)."""
        return (np.arange(1, self.ns) - 0.5) / (self.ns - 1)

    def field(self, name: str, surface_index: int = -1) -> ToroidalField:
        """Read one spectral field at its native file row.

        Names: r, z [m], lambda [rad], b [T], jacobian [m^3],
        bsup_theta/bsup_zeta [T/m], bsub_theta/bsub_zeta [T m], bsub_s [T m].
        Magnetic components are contravariant/covariant components in
        (s, theta, zeta), not orthonormal cylindrical components. Jacobian
        is the signed coordinate Jacobian, retaining VMEC's signgs.
        Geometry and bsub_s use the full grid; all others use the half grid.
        The two grids' identical row numbers denote different flux surfaces.
        """
        cosine, sine, grid, dim, sine_symmetric = _FIELDS[name]
        index = _surface_index(surface_index, self.ns)
        if grid == "half" and index == 0:
            raise ValueError(f"{name} has no physical half-grid value at file row 0")
        return _read_fourier(
            self._data, self._modes[dim], cosine, sine, ("radius", dim), index,
            lasym=self.lasym, sine_symmetric=sine_symmetric,
        )

    def surface(self, surface_index: int = -1) -> CylindricalSurface:
        """Return the full-grid surface in metres, with physical phi=zeta."""
        return CylindricalSurface(
            self.field("r", surface_index), self.field("z", surface_index)
        )

    def profile(
        self,
        name: str,
        *,
        grid: Literal["full", "half"] | None = None,
        unit: str | None = None,
    ) -> RadialProfile:
        """Read a scalar radial variable on normalized toroidal flux s.

        Standard profiles use an explicit grid table. Other radius variables
        require grid='full' or 'half'; grid cannot override a known convention.
        Half-grid row 0 is omitted. Units are copied from file metadata unless
        explicitly supplied; values are never rescaled.
        """
        known_grid = _PROFILE_GRIDS.get(name)
        if grid is None:
            grid = known_grid
        if grid not in ("full", "half"):
            raise ValueError(f"specify grid='full' or 'half' for {name}")
        if known_grid is not None and grid != known_grid:
            raise ValueError(f"{name} is defined on the {known_grid} grid")
        values = self._data.array(
            name, ("radius",), index=slice(1, None) if grid == "half" else None,
        )
        coordinate = self.s_full if grid == "full" else self.s_half
        return RadialProfile(
            coordinate, values, name=name,
            unit=self._data.units[name] if unit is None else unit, grid=grid,
        )
