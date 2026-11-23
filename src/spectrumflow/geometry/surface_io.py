"""Version 1 NPZ surface schema; coefficients follow ToroidalField directly."""

from __future__ import annotations

from os import PathLike

import numpy as np

from ..toroidalfield import ToroidalField
from .surface import BoozerSurface, CylindricalSurface, Surface

_FORMAT = "spectrumflow.surface"
_PHASE = "exp(i*(m*theta-n*nfp*zeta))"


def save_surface(surface: Surface, path: str | PathLike[str]) -> None:
    """Internal writer, with no filename extension rewriting or object arrays."""
    if type(surface) not in (CylindricalSurface, BoozerSurface):
        raise TypeError("only CylindricalSurface and BoozerSurface can be saved")
    data = {
        "format": np.array(_FORMAT),
        "version": np.array(1),
        "coordinates": np.array("boozer" if isinstance(surface, BoozerSurface) else "cylindrical"),
        "phase": np.array(_PHASE),
        "length_unit": np.array("m"),
        "angle_unit": np.array("rad"),
        "nfp": np.array(surface.nfp),
        "toroidal_sign": np.array(surface.toroidal_sign),
        **{f"{name}_coefficients": value.coefficients for name, value in surface._fields.items()},
    }
    # A file handle makes np.savez preserve the exact user-supplied filename.
    with open(path, "wb") as stream:
        np.savez_compressed(stream, **data)


def load_surface(path: str | PathLike[str]) -> Surface:
    """Internal reader: reject unknown schemas instead of guessing conventions."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "format", "version", "coordinates", "phase", "length_unit",
            "angle_unit", "nfp", "toroidal_sign", "r_coefficients", "z_coefficients",
        }
        if not required.issubset(data.files):
            raise ValueError("incomplete surface archive")

        def scalar(name: str, kind: str):
            value = data[name]
            if value.shape != () or value.dtype.kind not in kind:
                raise ValueError(f"invalid surface metadata: {name}")
            return value.item()

        for name, expected in (
            ("format", _FORMAT), ("phase", _PHASE),
            ("length_unit", "m"), ("angle_unit", "rad"),
        ):
            if scalar(name, "U") != expected:
                raise ValueError(f"unsupported surface {name}")
        if scalar("version", "iu") != 1:
            raise ValueError("unsupported surface version")
        coordinates = scalar("coordinates", "U")
        if coordinates not in ("cylindrical", "boozer"):
            raise ValueError("unsupported surface coordinates")
        if coordinates == "boozer":
            required.add("omega_coefficients")
        if set(data.files) != required:
            raise ValueError("unexpected or missing surface arrays")
        nfp = scalar("nfp", "iu")
        sign = scalar("toroidal_sign", "iu")
        fields = {
            name: ToroidalField(data[f"{name}_coefficients"], nfp=nfp)
            for name in (("r", "z", "omega") if coordinates == "boozer" else ("r", "z"))
        }
    cls = BoozerSurface if coordinates == "boozer" else CylindricalSurface
    return cls(**fields, toroidal_sign=sign)
