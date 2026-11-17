"""Check axes ownership, physical angle ranges and plotting side effects."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrumflow import ToroidalField
from spectrumflow.toroidalfield import plot_field


@pytest.mark.parametrize("one_period,filled", [(True, True), (False, False)])
def test_plot_returns_artists_and_correct_angle_ranges(one_period, filled):
    field = ToroidalField.from_modes({(1, 1): 0.5}, nfp=4)
    fig, ax = plt.subplots()
    original_style = dict(matplotlib.rcParams)
    returned, artist = plot_field(field, ax=ax, one_period=one_period, filled=filled, levels=8)
    assert returned is ax
    assert artist.axes is ax
    np.testing.assert_allclose(ax.get_xlim(), [0, 2*np.pi/(4 if one_period else 1)], rtol=0, atol=1e-14)
    np.testing.assert_allclose(ax.get_ylim(), [0, 2*np.pi], rtol=0, atol=1e-14)
    assert "rad" in ax.get_xlabel() and "rad" in ax.get_ylabel()
    assert len(fig.axes) == 1
    assert dict(matplotlib.rcParams) == original_style
    fig.canvas.draw()
    plt.close(fig)


def test_plot_creates_axes_for_constant_field():
    ax, artist = plot_field(ToroidalField.constant(2))
    assert artist.axes is ax
    ax.figure.canvas.draw()
    plt.close(ax.figure)
