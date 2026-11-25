# spectrumflow

个人 Python 科研数据处理包，服务于仿星器研究。

目前提供 `toroidalfield` 实标量谱场、`geometry` 环面与度规计算、
VMEC / booz_xform NetCDF 读取，以及标量径向数据类 `RadialProfile`。

## 安装与构建

要求 Python >= 3.10。

```bash
python -m pip install .
python -m pip install -e '.[dev]'    # 本地开发
python -m pip install '.[netcdf]'    # 读取 wout / boozmn 文件
python -m pytest
python -m build
```

## 快速使用

### 环形场

```python
import numpy as np
from spectrumflow import ToroidalField

field = ToroidalField.from_modes({(0, 0): 1.0, (1, 0): 0.2}, nfp=1)
theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
values = field(theta, 0.0)
```

### 环面

`CylindricalSurface(r, z)` 使用柱坐标角参数化，`BoozerSurface(r, z, omega)`
使用 Boozer 角参数化。`r`、`z`、`omega` 都是 `ToroidalField`，必须共享 `nfp`
和参数角定义，允许各自保留不同的谱分辨率；每个对象表示一个二维曲面，不包含径向网格。

物理柱坐标角统一定义为 `phi = toroidal_sign*zeta + omega(theta, zeta)`，
柱坐标表示取 `omega=0`。`toroidal_sign` 默认为 `+1`，也可显式设为 `-1`；
需要相反的 omega 符号时传入 `-omega`。这一约定对应 tfpy 的
`toroidal_sign = -1 if reverseToroidalAngle else 1`，并在
`reverseOmegaAngle=True` 时将原 omega 取负。R、Z 始终在传入的参数角求值。
笛卡尔坐标使用右手系 `(R*cos(phi), R*sin(phi), Z)`。
Boozer 角的磁坐标含义由输入平衡数据确定，几何类本身不施加磁场约束。

```
import numpy as np
from spectrumflow import ToroidalField
from spectrumflow.geometry import CylindricalSurface

theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)

# 大半径 3 m、小半径 0.7 m 的圆环面；沿用 ToroidalField 的系数约定。
r = ToroidalField.from_modes({(0, 0): 3, (1, 0): 0.35}, nfp=3)
z = ToroidalField.from_modes({(1, 0): -0.35j}, nfp=3)
surface = CylindricalSurface(r, z)
xyz = surface.position(theta, 0.0)       # (100, 3)，最后一轴是 x, y, z
gtt, gtz, gzz = surface.metric()         # 三个 ToroidalField
g = surface.metric_tensor(theta, 0.0)    # (100, 2, 2)，轴序 theta, zeta
area, volume = surface.area_volume()

print(f"位置数组形状：{xyz.shape}，度规张量形状：{g.shape}")
print(f"面积：{area:.8f} m²，体积：{volume:.8f} m³")
```

角度输入遵循 NumPy 广播：两个一维数组按点配对，使用
`theta[:, None]`、`zeta[None, :]` 得到轴序为 `(theta, zeta)` 的张量网格。
`phi()` 返回不折叠的物理角，`rz()` 返回两个广播后的数组。
`basis()` 返回笛卡尔协变基矢，形状为 `(..., 2, 3)`；
`covariant_basis()` 则返回两组三分量谱场，分量相对于局部柱坐标**正交单位基**
`(e_R, e_phi, e_Z)`，其角向分量包含 R 因子。
度规为 `g_ab = R_a*R_b + R²*phi_a*phi_b + Z_a*Z_b`。
谱乘积保留完整卷积，分辨率可能增长；仅调用 `with_resolution(mpol, ntor)`
才显式截断或补零，并返回新对象。

`area_volume(ntheta=129, nzeta=129)` 在不含重复端点的单周期网格上积分，
结果包含全部 `nfp` 个周期。曲面须正则、R>0 且无自交；面积通常不是有限谱，
应提高采样数检查收敛。环向参数增加 `2*pi/nfp` 时，位置旋转一个场周期，
R、Z 和度规周期重复。

安装 `spectrumflow[plot]` 后，两种环面均可绘制固定物理柱坐标角的 R–Z 截面：

```python
ax, lines = surface.plot_cross_section()  # 默认 phi=[0, np.pi/surface.nfp]
ax, lines = surface.plot_cross_section(
    phi=[0, 0.2], ntheta=361, fontsize=17, usetex=True,
)
```

`phi` 单位为弧度，可传单个角或一维角数组；所有截面叠加在同一等比例坐标轴上，
返回轴及按输入顺序排列的曲线。默认不添加标题，轴标签为 `$R$`、`$Z$`，
不显示单位；标签、刻度为 17 号字，刻度数字使用数学模式，并调用外部 LaTeX
渲染（需本机安装 LaTeX）；`usetex=False` 可使用 Matplotlib 内置渲染。
支持传入 `ax`、`figsize`、`tick_fontsize` 以及 `color`、`linewidth` 等曲线参数，
不修改全局样式，不自动显示或保存。
Boozer 环面会反解 `phi = toroidal_sign*zeta + omega`，要求固定 theta 时
phi 随 zeta 严格单调、没有折叠；不会将固定 Boozer 角当作物理截面。
`surface.cross_section(theta, phi)` 可独立计算截面的 `(R, Z)`，输入遵循
NumPy 广播，两个输出均为广播后的形状；与 `rz(theta, zeta)` 的角度含义不同。

两种环面也可绘制三维曲面，返回三维坐标轴和可继续调整的曲面 artist：

```python
ax, artist = surface.plot_3d()  # 默认绘制完整环面
ax, artist = surface.plot_3d(
    one_period=True, ntheta=129, nzeta=129,
    fontsize=17, usetex=True, color="steelblue", alpha=0.8,
)
```

`plot_3d()` 使用 `position()` 计算笛卡尔坐标，保留 Boozer 的 omega 修正和环向
方向约定。默认参数角范围为 theta、zeta 均在 `[0, 2*pi]`；`one_period=True`
将 zeta 范围改为 `[0, 2*pi/nfp]`，Boozer 曲面的周期边缘不一定是固定物理 phi。
含重复端点的网格轴序为 `(theta, zeta)`，默认采样数为 `(129, 257)`。
默认绘制全部采样网格，可通过 `rcount`、`ccount` 或 `rstride`、`cstride`
显式降低绘图分辨率；应提高采样数检查几何收敛。
三个坐标轴保持相同物理比例，标签为 `$x$`、`$y$`、`$z$`，不显示单位、不加标题；
坐标数据仍为米。默认标签和刻度为 17 号字，含刻度数字的文字使用 LaTeX 渲染。
支持 `usetex=False`、`tick_fontsize`、`figsize` 和已有的三维 `ax`，其余参数传给
Matplotlib 的 `plot_surface()`；不修改全局样式，不自动显示或保存。

### 径向数据

一个 `RadialProfile` 存储一个物理量，显式保留径向坐标、名称和单位。
输入和返回的数组均为独立副本，轴序为 `(nr,)`；坐标必须有限且严格递增，
数据必须是有限实数。默认径向坐标为归一化环向磁通 `s`。

```python
from spectrumflow import RadialProfile

pressure = RadialProfile(
    [0, 0.25, 0.5, 1], [1000, 750, 500, 0], name="pressure", unit="Pa",
)
print(pressure(0.4))                 # 线性插值；保持输入形状，不允许外插
ax, line = pressure.plot(color="C1") # 需要 [plot]，返回轴和 Line2D
```

其他径向坐标通过 `coordinate_name`、`coordinate_unit` 声明；不自动进行
`s`、`sqrt(s)` 与米制半径之间的转换。`.coordinate` 和 `.values` 返回原始采样，
`.grid` 可记录 `"full"` / `"half"` 等来源信息。绘图标签包含给定名称和单位，
接受 `ax`、`xlabel`、`ylabel` 及 Matplotlib 曲线参数，不显示或保存文件。

### VMEC 与 booz_xform 输出

安装 `[netcdf]` 后可读取标准 NetCDF `wout_*.nc` 和 `boozmn_*.nc` 文件。
构造时读入内存并关闭文件；导入核心包或读取器本身不加载 NetCDF、Matplotlib、
xarray 或 booz_xform 计算包。`variables` 列出原始变量，`output["变量名"]`
返回保留文件轴序、单位和缺失值掩码的数组副本。

```python
from spectrumflow.VMEC import VmecOutput
from spectrumflow.booz_xform import BoozXformOutput

vmec = VmecOutput("wout_case.nc")
surface = vmec.surface()                    # 最外层全网格曲面，phi=zeta
b = vmec.field("b")                        # 最外层半网格 |B|，ToroidalField
lam = vmec.field("lambda")                 # theta*=theta+lambda
iota = vmec.profile("iotas")                # 半网格，已去掉占位行
pressure = vmec.profile("presf")            # 全网格，单位保留文件元数据

booz = BoozXformOutput("boozmn_case.nc")
surface_b = booz.surface()                  # 最后一个 jlist 曲面
b_b = booz.field("b")
print(booz.s)                              # 已转换曲面的实际归一化磁通
print(booz.profile("iota_b")(booz.s))       # 在这些半网格位置取 iota
```

两种读取器都根据文件的 `lasym__logical__` 判断对称性；非对称输出必须提供
对应的正弦、余弦分量。通过实际模态编号转换系数，保留所有模态，零模计一次，
其余系数遵循 `ToroidalField` 的归一化。文件的 `xn` / `ixn_b` 已含 `nfp`，
读取时显式除去该因子；不依赖模态排序或零模位置，也不反转环向角。

VMEC `field(name, surface_index=-1)` 的物理量如下。索引使用文件径向行号，
允许负索引；半网格行号 0 为无物理意义的占位行，访问会报错。

| name | 径向网格 | 含义及单位 |
| --- | --- | --- |
| `r`, `z` | 全网格 | 柱坐标 R、Z [m] |
| `lambda` | 半网格 | 直磁力线极向角修正 [rad] |
| `b` | 半网格 | 磁场强度 [T] |
| `jacobian` | 半网格 | `(s, theta, zeta)` 有符号 Jacobian [m³] |
| `bsup_theta`, `bsup_zeta` | 半网格 | VMEC 坐标逆变磁场分量 [T/m] |
| `bsub_theta`, `bsub_zeta` | 半网格 | VMEC 坐标协变磁场分量 [T m] |
| `bsub_s` | 全网格 | 径向协变磁场分量 [T m] |

VMEC `s_full[j]=j/(ns-1)`；半网格文件行 `j=1,...,ns-1` 对应
`s=(j-0.5)/(ns-1)`，`.s_half` 不包含占位行。因此同一行号的几何、磁场
不在同一个径向位置，读取器不自动插值来匹配它们。`profile()` 对常见物理量
使用明确的网格表，例如 `iotaf` / `presf` 是全网格，`iotas` / `pres` 是半网格；
其他径向变量必须显式传 `grid="full"` 或 `grid="half"`，已知约定不允许覆盖。
所有 profile 的坐标都是 `s`，其 `.grid` 标明原始网格；`unit=` 只覆盖标签，
不缩放数值。Jacobian 和磁场分量保留文件方向、符号和基底。
网格与基底约定可参阅 [VMEC 后处理说明](https://proximafusion.github.io/vmecpp/tutorials/postprocessing_wout.html)。

booz_xform 的 `field()` 支持 `r`、`z`、`omega`、`b`、`jacobian`。
索引始终指向按 `jlist` 顺序打包的谱曲面；`.ns` 是原 VMEC 全网格点数
（文件 `ns_b`），`.surface_count` 是转换曲面数。文件的 `jlist` 为从 1 开始
的 VMEC 行号，对应半网格位置 `s=(jlist-1.5)/(ns-1)`，最后一个曲面通常
也不在 `s=1`。文件 `pmns_b` / `pmnc_b` 存储 `p=-nu=phi-zeta_B`，
直接作为几何的 `omega`，即 `phi=zeta_B+omega`。
`jacobian` 来自 `gmn_b` / `gmns_b`，是 `(psi, theta_B, zeta_B)` 的 Jacobian，
`psi` 为环向磁通除以 `2*pi`，单位 [Wb]；Jacobian 单位为 [m/T]。
相关定义见 [booz_xform API](https://hiddensymmetries.github.io/booz_xform/api.html)
及 [文件变量约定](https://booz-xform-jax.readthedocs.io/en/latest/inputs_outputs.html)。

`booz.profile()` 读取原始径向数组，`phi_b` 在全网格，
`iota_b`、`buco_b`、`bvco_b`、`pres_b`、`beta_b`、`phip_b` 在去掉占位行的
完整半网格，而非仅限于 `jlist`。例如打包曲面 k 的 iota 可以直接取
`booz.profile("iota_b").values[booz.jlist[k]-2]`。目前支持标准 NetCDF 文件，
不读取旧 Fortran 无格式二进制输出，也不运行坐标变换计算。

## 许可证

Copyright (c) 2026 Ke Liu。项目采用 GNU General Public License v3.0，
详见 [LICENSE](LICENSE)。
