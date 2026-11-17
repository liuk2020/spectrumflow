# spectrumflow

个人 Python 科研数据处理与科技绘图包，服务于仿星器研究。

当前仅包含源码结构和类占位符，尚未实现数值计算、文件读取或绘图功能。

| 源码文件（相对 `src/spectrumflow/`） | 类占位符 | 职责 |
| --- | --- | --- |
| `toroidalfield/field.py` | `ToroidalField` | 双角标量场的 Fourier 谱表示与数值运算 |
| `geometry/surface.py` | `Surface` | 环面曲面表示与几何量计算 |
| `geometry/curve.py` | `Curve` | 空间曲线表示与几何量计算 |
| `current/surface_current.py` | `SurfaceCurrent` | 环面上的面电流分布 |
| `current/line_current.py` | `LineCurrent` | 沿空间曲线的线电流 |
| `VMEC/output.py` | `VmecOutput` | VMEC 输出读取、约定转换与后处理 |
| `SPEC/output.py` | `SpecOutput` | SPEC 输出读取、约定转换与后处理 |
