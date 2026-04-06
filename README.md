# HongMeng Data Process Tools

鸿蒙计划高频频谱仪数据处理工具集。

## 文件说明

| 文件 | 说明 |
| ---- | ---- |
| `HongMeng_raw_data_Parser.py` | 原始数据解析器 — 支持 SPEC/VNA/TEMP 三种数据类型解包 |
| `HongMeng_Preprocessor.py` | 数据预处理（开关矩阵标定等）【开发中】|
| `Calibration_tools.py` | 校准工具函数库 |

## 快速开始

```python
from HongMeng_raw_data_Parser import HongMengFileProcessor

processor = HongMengFileProcessor()
result = processor.process_file('data.dat', save=True)

# SPEC 相关器数据
spec_data = result['spec']['data']           # (n_fft, 4, 4096) int64

# VNA S参数
s11 = result['vna']['data']['s11'][0]        # 第 0 次扫频 S11 复数值
mag_dB = 20 * np.log10(np.abs(s11))          # |S11| in dB

# 温度
temp_raw = result['temp']['raw']             # (n_pkt,) 75-byte 原始数据
```

## 文档

- [Parser 详细文档](docs/README_HongMeng_raw_data_Parser.md)
- [操作手册](docs/DSL_TestPlatform%20operating%20handbook.md)
- [result 数据结构](docs/result_structure.txt)
- [更新日志](docs/CHANGELOG.md)

## 环境要求

- Python 3.9+
- numpy >= 1.20.0
