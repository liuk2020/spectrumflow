# spectrumflow

个人 Python 科研数据处理包，服务于仿星器研究。

目前提供 `toroidalfield` 实标量谱场，以及 `geometry` 中的环面表示与度规计算。

## 安装与构建

要求 Python >= 3.10。

```bash
python -m pip install .
python -m pip install -e '.[dev]'    # 本地开发
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
和参数角定义，允许各自保留不同的谱分辨率；`r`、`z` 的单位为米，角度为弧度。
每个对象表示一个二维曲面，不包含径向网格。

物理柱坐标角统一定义为 `phi = toroidal_sign*zeta + omega(theta, zeta)`，
柱坐标表示取 `omega=0`。`toroidal_sign` 默认为 `+1`，也可显式设为 `-1`；
需要相反的 omega 符号时传入 `-omega`。这一约定对应 tfpy 的
`toroidal_sign = -1 if reverseToroidalAngle else 1`，并在
`reverseOmegaAngle=True` 时将原 omega 取负。R、Z 始终在传入的参数角求值。
笛卡尔坐标使用右手系 `(R*cos(phi), R*sin(phi), Z)`。
Boozer 角的磁坐标含义由输入平衡数据确定，几何类本身不施加磁场约束。

```python
from spectrumflow.geometry import CylindricalSurface, BoozerSurface, Surface

# 大半径 3 m、小半径 0.7 m 的圆环面；沿用 ToroidalField 的系数约定。
r = ToroidalField.from_modes({(0, 0): 3, (1, 0): 0.35}, nfp=3)
z = ToroidalField.from_modes({(1, 0): -0.35j}, nfp=3)
surface = CylindricalSurface(r, z)
xyz = surface.position(theta, 0.0)       # (100, 3)，最后一轴是 x, y, z
gtt, gtz, gzz = surface.metric()         # 三个 ToroidalField
g = surface.metric_tensor(theta, 0.0)    # (100, 2, 2)，轴序 theta, zeta
area, volume = surface.area_volume()    # 完整环面的 m²、m³

omega = ToroidalField.from_modes({(1, 1): -0.02j}, nfp=3)
boozer = BoozerSurface(r, z, omega, toroidal_sign=-1)
boozer.save("surface.npz")
restored = Surface.load("surface.npz")  # 自动恢复为 BoozerSurface
```

角度输入遵循 NumPy 广播：两个一维数组按点配对，使用
`theta[:, None]`、`zeta[None, :]` 得到轴序为 `(theta, zeta)` 的张量网格。
`phi()` 返回不折叠的物理角，`rz()` 返回两个广播后的数组。
`basis()` 返回笛卡尔协变基矢，形状为 `(..., 2, 3)`；
`covariant_basis()` 则返回两组三分量谱场，分量相对于局部柱坐标**正交单位基**
`(e_R, e_phi, e_Z)`，其角向分量包含 R 因子。
度规为 `g_ab = R_a*R_b + R²*phi_a*phi_b + Z_a*Z_b`，单位为 m²/rad²。
谱乘积保留完整卷积，分辨率可能增长；仅调用 `with_resolution(mpol, ntor)`
才显式截断或补零，并返回新对象。

`area_volume(ntheta=129, nzeta=129)` 在不含重复端点的单周期网格上积分，
结果包含全部 `nfp` 个周期。曲面须正则、R>0 且无自交；面积通常不是有限谱，
应提高采样数检查收敛。环向参数增加 `2*pi/nfp` 时，位置旋转一个场周期，
R、Z 和度规周期重复。

IO 使用压缩 NPZ，无新增依赖、不使用 pickle，不自动修改文件名。
版本 1 保存格式标识、版本、坐标类型、相位约定、单位、`nfp`、`toroidal_sign`
以及各场完整的 `(m, n)` 复系数数组；分辨率由数组形状确定。
读取时检查元数据、系数形状和共轭对称性。`Surface.load()` 自动识别类型，
`CylindricalSurface.load()`、`BoozerSurface.load()` 同时检查目标类型。

## 许可证

Copyright (c) 2026 Ke Liu。项目采用 GNU General Public License v3.0，
详见 [LICENSE](LICENSE)。
