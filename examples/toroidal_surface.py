"""Reproducible torus metric and NPZ example; no external equilibrium data."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_allclose

from spectrumflow import ToroidalField
from spectrumflow.geometry import BoozerSurface, CylindricalSurface, Surface


def main() -> None:
    r = ToroidalField.from_modes({(0, 0): 3, (1, 0): 0.35}, nfp=3)
    z = ToroidalField.from_modes({(1, 0): -0.35j}, nfp=3)
    cylinder = CylindricalSurface(r, z)
    omega = ToroidalField.from_modes({(1, 1): -0.02j}, nfp=3)
    boozer = BoozerSurface(r, z, omega, toroidal_sign=-1)

    theta = np.linspace(0, 2*np.pi, 17, endpoint=False)[:, None]
    zeta = np.linspace(0, 2*np.pi/3, 13, endpoint=False)[None, :]
    for surface in (cylinder, boozer):
        tensor = surface.metric_tensor(theta, zeta)
        for field, (a, b) in zip(surface.metric(), ((0, 0), (0, 1), (1, 1))):
            assert_allclose(field(theta, zeta), tensor[..., a, b], rtol=1e-12, atol=1e-12)
        area, volume = surface.area_volume(33, 33)
        assert_allclose(area, 4*np.pi**2*3*0.7, rtol=1e-12)
        assert_allclose(volume, 2*np.pi**2*3*0.7**2, rtol=1e-12)
        print(f"{type(surface).__name__}: area={area:.8f} m², volume={volume:.8f} m³")

    with TemporaryDirectory() as directory:
        path = Path(directory) / "surface.npz"
        boozer.save(path)
        restored = Surface.load(path)
        assert_allclose(restored.position(theta, zeta), boozer.position(theta, zeta), rtol=0, atol=0)
        print(f"NPZ round trip: {type(restored).__name__}, nfp={restored.nfp}")


if __name__ == "__main__":
    main()
