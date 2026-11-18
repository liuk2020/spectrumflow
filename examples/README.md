# 示例

安装 `spectrumflow[plot]` 后运行 `python examples/toroidal_field.py`，
演示解析场构造、FFT 重建、微分与绘图，无需外部数据。
参照 tfpy 案例，在同一坐标轴上叠加两种绘图：`field1` 通过原函数在单周期、
不含重复端点的网格上采样后调用 `from_samples()` 构造，用不填充的等值线显示
整个 $[0,2\pi]\times[0,2\pi]$ 角度域；`field_2` 通过 `from_modes()` 构造，
用填充等值图显示一个环向周期。两者均调用 `.plot()`，第二次绘图显式复用
第一次返回的 `contours.levels`，并使用相同的默认配色。叠加后将横轴范围和刻度
统一设为全域的 0、π、2π。

`ToroidalField.plot()` 默认使用 tfpy 风格的 `tol:sunset` 配色、π 刻度和 18 号字。
横轴始终是实际参数角 ζ [rad]；单周期的刻度标签为
0、$\pi/N_\mathrm{fp}$、$2\pi/N_\mathrm{fp}$，刻度位置使用实际 nfp 计算。
`one_period=False` 绘制完整环向范围，刻度为 0、π、2π。

```python
ax, artist = field.plot(
    cmap="tol:sunset", levels=30, filled=True,
    fontsize=18, tick_fontsize=16,
    xlabel=r"$\zeta$ [rad]", ylabel=r"$\theta$ [rad]",
    usetex=False, figsize=(7, 5),
    colorbar=True, colorbar_kwargs={"label": "f [a.u.]"},
)
```

`filled=False` 使用等值线；可用 `cmap="viridis"` 等 Matplotlib 配色，
或通过 `colors="black"` 指定线条颜色。`levels`、`vmin`、`vmax`、
`linewidths` 等参数直接传给 Matplotlib。`usetex=True` 开启外部 LaTeX
渲染（需本机安装 LaTeX）；`usetex=False` 使用内置数学公式渲染。
色条默认关闭，开启后可通过 `artist.colorbar` 继续调整。
传入 `ax` 时使用其所属 figure，`figsize` 仅在创建新 axes 时生效。
方法不修改全局样式，也不调用 `show()` 或保存文件。
