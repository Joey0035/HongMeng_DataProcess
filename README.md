# HongMeng Data Process Tools — v4.0

鸿蒙计划高频频谱仪数据处理工具集。

## 文件说明

| 文件 | 说明 |
| ---- | ---- |
| `HongMeng_raw_data_Parser.py` | 原始数据解析器 — 支持 SPEC/VNA/TEMP 三种数据类型解包 |
| `HongMeng_DataInspector.py` | 数据检视工具 — `EffectiveDataExtractor` / `DataInspector`：按源分割、打印概览、绘制频谱/S11面板 |
| `HongMeng_VNA_Calibrator.py` | VNA OSL 校准引擎 — 1-port 3-term 误差模型，支持开关面/LNA面双校准平面 |
| `HongMeng_Preprocessor.py` | 数据预处理（按观测序列分割等）|
| `dat_to_npz.py` | 命令行批量转换工具 — 将超大 `.dat` 文件解包并保存为 `.npz` |
| `streamlit_app/` | Streamlit 数据监视平台 — 包数据状态 / 温度 / 频谱 / S参数四个页面 |
| `Calibration_tools.py` | 校准工具函数库（旧版） |

## 快速开始

### 解析原始数据

```python
from HongMeng_raw_data_Parser import HongMengFileProcessor

processor = HongMengFileProcessor()
result = processor.process_file('data.dat', save=True)

# SPEC 相关器数据
spec_data = result['spec']['data']           # (n_fft, 4, 4096) int64

# VNA S参数
s11 = result['vna']['data'][0]              # 第 0 次扫频 S11 复数值
mag_dB = 20 * np.log10(np.abs(s11))         # |S11| in dB

# 温度 (5 chips × 5 channels)
temp_data = result['temp']['data']          # (n_pkt, 5, 5) float64, ℃
obs_seq   = result['spec']['obs_seq']       # 检测到的观测序列，如 [30,31,...,45,23]
```

### 数据检视

```python
from HongMeng_DataInspector import DataInspector

insp = DataInspector(result, vna_freq={901: (30, 120), 1901: (1, 190)})
insp.info()                          # 打印数据概览
insp.plot_spec_panel('1')            # SPEC 频谱面板（Auto1）
insp.plot_vna_panel(n_freq=901)      # VNA S11 面板（901点扫频）
```

### VNA OSL 校准

```python
from HongMeng_DataInspector import DataInspector
from HongMeng_VNA_Calibrator import calibrate_vna_data
import numpy as np

insp = DataInspector(result, vna_freq={901: (30, 120)})
split = insp.split_vna_by_src(n_freq=901)
freq_hz = np.linspace(30e6, 120e6, 901)

calibrated, diagnostics = calibrate_vna_data(split, freq_hz)
```

### Noisewave校准
```python
from calibration.workflow import CalibrationWorkflowConfig, run_calibration_workflow

cfg = CalibrationWorkflowConfig(
    vna_freq_hz=np.linspace(30e6, 200e6, 1701),
    spec_freq_range_mhz=(50, 180),
    receiver_cable_snp="./MWS_dianxing_cable_S/33333-1.s2p",
    cal_sources = (
        'LgO', 'LgS', 'ShtS', 'ShtR1', 'ShtR2',
        'R3', 'R4', 'R5', 'Cal_O', 'Cal_S',
    ),
    noise_source_terms = (7, 9),
    noise_wave_fit_term = 7,
    noise_wave_iterations = 10,
    use_legacy_vna_cal = False
)

cal = run_calibration_workflow(result, cfg)

a = cal.model
temps = cal.recovered_temperatures
vna_by_src = cal.vna_by_src
```

### 命令行转换

```bash
# 解包 .dat 并保存为 .npz
python dat_to_npz.py /data/obs_20260401.dat

# 指定输出路径
python dat_to_npz.py /data/obs_20260401.dat /out/obs.npz
```

### 启动 Streamlit 监视平台

```bash
cd streamlit_app
streamlit run app.py
```

## 文档

- [Parser 详细文档](docs/README_HongMeng_raw_data_Parser.md)
- [操作手册](docs/DSL_TestPlatform%20operating%20handbook.md)
- [result 数据结构](docs/result_structure.txt)
- [更新日志](docs/CHANGELOG.md)

## 环境要求

- Python 3.9+
- numpy >= 1.20.0
- matplotlib
- streamlit（仅 `streamlit_app/` 需要，见 `streamlit_app/requirements.txt`）
