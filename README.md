# spectrumflow

个人 Python 科研数据处理包，服务于仿星器研究。

目前提供 `toroidalfield`，用于表示和计算环面上的实标量场。

## 安装与构建

要求 Python >= 3.10。

```bash
python -m pip install .
python -m pip install -e '.[dev]'    # 本地开发
python -m pytest
python -m build
```

## 快速使用

```python
import numpy as np
from spectrumflow import ToroidalField

field = ToroidalField.from_modes({(0, 0): 1.0, (1, 0): 0.2}, nfp=1)
theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
values = field(theta, 0.0)
```

## 许可证

Copyright (c) 2026 Ke Liu。项目采用 GNU General Public License v3.0，
详见 [LICENSE](LICENSE)。
