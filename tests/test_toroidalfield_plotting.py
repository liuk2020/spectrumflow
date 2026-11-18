"""Check axes ownership, physical angle ranges and plotting side effects."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("cmap")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrumflow import ToroidalField


@pytest.mark.parametrize("one_period,filled", [(True, True), (False, False)])
def test_plot_returns_artists_and_correct_angle_ranges(one_period, filled):
    field = ToroidalField.from_modes({(1, 1): 0.5}, nfp=4)
    fig, ax = plt.subplots()
    original_style = dict(matplotlib.rcParams)
    returned, artist = field.plot(
        ax=ax, one_period=one_period, filled=filled, levels=8, usetex=False,
    )
    assert returned is ax
    assert artist.axes is ax
    assert artist.filled is filled
    assert artist.cmap.name == "tol:sunset"
    np.testing.assert_allclose(ax.get_xlim(), [0, 2*np.pi/(4 if one_period else 1)], rtol=0, atol=1e-14)
    np.testing.assert_allclose(ax.get_ylim(), [0, 2*np.pi], rtol=0, atol=1e-14)
    assert ax.get_xlabel() == r"$\zeta$" and ax.get_ylabel() == r"$\theta$"
    np.testing.assert_allclose(ax.get_xticks(), [0, np.pi/(4 if one_period else 1), 2*np.pi/(4 if one_period else 1)], rtol=0, atol=1e-14)
    expected_labels = ["$0$", r"$\pi/N_\mathrm{fp}$", r"$2\pi/N_\mathrm{fp}$"] if one_period else ["$0$", r"$\pi$", r"$2\pi$"]
    assert [label.get_text() for label in ax.get_xticklabels()] == expected_labels
    assert len(fig.axes) == 1
    assert dict(matplotlib.rcParams) == original_style
    fig.canvas.draw()
    plt.close(fig)


def test_plot_creates_axes_for_constant_field():
    ax, artist = ToroidalField.constant(2).plot(usetex=False)
    assert artist.axes is ax
    ax.figure.canvas.draw()
    plt.close(ax.figure)


@pytest.mark.parametrize("one_period", [True, False])
def test_plot_uses_actual_angles_for_coordinates_and_field(monkeypatch, one_period):
    field = ToroidalField.from_modes({(1, 1): 0.5, (2, -1): -0.25j}, nfp=3)
    fig, ax = plt.subplots()
    contourf = ax.contourf
    captured = {}

    def capture(x, y, values, **kwargs):
        captured.update(x=x, y=y, values=values)
        return contourf(x, y, values, **kwargs)

    monkeypatch.setattr(ax, "contourf", capture)
    field.plot(ax=ax, ntheta=17, nzeta=23, one_period=one_period, usetex=False)
    x, y = captured["x"], captured["y"]
    np.testing.assert_allclose(x[[0, -1]], [0, 2*np.pi/(3 if one_period else 1)], rtol=0, atol=1e-14)
    expected = np.cos(y[:, None] - 3*x[None, :]) + 0.5*np.sin(2*y[:, None] + 3*x[None, :])
    np.testing.assert_allclose(captured["values"], expected, rtol=1e-12, atol=1e-14)
    assert captured["values"].shape == (17, 23)
    labels = ["$0$", r"$\pi/N_\mathrm{fp}$", r"$2\pi/N_\mathrm{fp}$"] if one_period else ["$0$", r"$\pi$", r"$2\pi$"]
    assert [label.get_text() for label in ax.get_xticklabels()] == labels
    np.testing.assert_allclose(ax.get_xticks(), [0, x[-1]/2, x[-1]], rtol=0, atol=1e-14)
    assert ax.get_xlabel() == r"$\zeta$"
    fig.canvas.draw()
    plt.close(fig)


def test_styling_and_colorbar_use_supplied_axes_figure():
    fig, ax = plt.subplots()
    other = plt.figure()
    original_style = dict(matplotlib.rcParams)
    _, artist = ToroidalField.from_modes({(1, 0): 0.5}).plot(
        ax=ax,
        cmap="plasma", xlabel="Toroidal angle [rad]", ylabel="Poloidal angle [rad]",
        fontsize=20, tick_fontsize=13, colorbar=True, usetex=False,
        colorbar_kwargs={"label": "f [a.u.]", "orientation": "horizontal"},
    )
    assert len(other.axes) == 0
    assert artist.cmap.name == "plasma"
    assert artist.colorbar.ax.figure is fig
    assert artist.colorbar.ax.get_xlabel() == "f [a.u.]"
    assert ax.get_xlabel() == "Toroidal angle [rad]"
    assert ax.get_ylabel() == "Poloidal angle [rad]"
    fig.canvas.draw()
    assert all(label.get_text().startswith("$") for label in artist.colorbar.ax.get_xticklabels())
    for styled in (ax, artist.colorbar.ax):
        assert styled.xaxis.label.get_fontsize() == 20
        assert all(label.get_fontsize() == 13 for label in styled.get_xticklabels())
    assert dict(matplotlib.rcParams) == original_style
    plt.close(fig)
    plt.close(other)


def test_explicit_line_colors_override_default_palette():
    ax, artist = ToroidalField.from_modes({(1, 0): 0.5}).plot(
        filled=False, usetex=False,
        colors="black", levels=[-0.5, 0, 0.5], linewidths=2, figsize=(7, 5),
    )
    assert not artist.filled
    np.testing.assert_allclose(artist.to_rgba(0), [0, 0, 0, 1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(ax.figure.get_size_inches(), [7, 5], rtol=0, atol=1e-14)
    ax.figure.canvas.draw()
    plt.close(ax.figure)


def test_tex_settings_survive_return_without_global_style_changes():
    original_style = dict(matplotlib.rcParams)
    ax, artist = ToroidalField.constant(2).plot(usetex=True, colorbar=True)
    for styled in (ax, artist.colorbar.ax):
        for axis in (styled.xaxis, styled.yaxis):
            assert axis.label.get_usetex()
            assert axis.get_offset_text().get_usetex()
            assert all(label.get_usetex() for label in axis.get_majorticklabels())
    assert dict(matplotlib.rcParams) == original_style
    plt.close(ax.figure)
