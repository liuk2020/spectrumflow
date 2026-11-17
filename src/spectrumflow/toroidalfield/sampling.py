"""Uniform angular grids and discrete Fourier transforms."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def angular_grid(ntheta: int, nzeta: int, nfp: int) -> tuple[NDArray, NDArray]:
    """Internal endpoint-excluded grid; counts and nfp are validated by callers."""
    return 2 * np.pi * np.arange(ntheta) / ntheta, 2 * np.pi * np.arange(nzeta) / (nfp * nzeta)


def coefficients_from_samples(values: ArrayLike) -> NDArray[np.complex128]:
    """Internal FFT in the package convention; see ToroidalField.from_samples."""
    values = np.asarray(values)
    if values.ndim != 2 or any(size % 2 != 1 for size in values.shape):
        raise ValueError("samples must be a 2D array with positive odd dimensions")
    if np.iscomplexobj(values):
        raise TypeError("samples must be real")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    # FFT uses positive synthesis frequencies on both axes; our n has the
    # opposite sign. Reversing the centered odd n axis converts that convention.
    spectrum = np.fft.fftshift(np.fft.fft2(values)) / values.size
    return spectrum[:, ::-1]
