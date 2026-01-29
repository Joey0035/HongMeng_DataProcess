# DSL 数据处理工具

DSL（Dark Sky Laboratory）FPGA 数据采集系统的原始数据解析与处理工具。

## 文件说明

- **HongMeng_raw_data_Parser.py** - 高性能原始数据解析器，支持 GB 级文件流式处理
- **Calibration_tools.py** - 校准工具，主要是高频使用函数
- **DSLpreprocess.py** - 数据预处理【开发中】

## 快速开始

### HongMeng_raw_data_Parser
```python
from unpack_DSLcorr_v3_optimized import run_ParceSpecPacket

result = run_ParceSpecPacket('data.dat', save=True)
```

详细文档见 [README_unpack_DSLcorr_v3.md](./docs/README_HongMeng_raw_data_Parser.md)

