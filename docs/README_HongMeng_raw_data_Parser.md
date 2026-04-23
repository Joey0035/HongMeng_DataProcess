# HongMeng Raw Data Parser

## 概述

`HongMeng_raw_data_Parser.py` — DSL 高频频谱仪原始 `.dat` 数据解包与处理程序。

**版本**: v4.0 | **作者**: JoeyXu | **日期**: 2026-04-23

---

## 已实现功能

- SPEC 相关器数据解包（4路自相关/互相关，72-bit → int64）
- VNA S参数解码（入射/反射 IQ → S11 线性复数值）
- TEMP 温度解码（offset-binary 24-bit → PT1000 CVD → ℃）
- 观测序列还原（RLE 检测 src 中的预设观测序列，输出 obs_seq）
- Checksum 逐包校验
- 不完整包自动剔除（头/尾截断数据识别与跳过）
- 伪同步码防御（`data_len` 上限检查）
- 数据异常自动检测（时间戳/丢包/VNA扫频/TEMP全NaN）
- 解包日志 `_parse.log`（丢弃包原因、位置、hex dump + 异常检测报告）
- 大文件流式读取（自动 / 手动阈值）
- NPZ 压缩存储

---

## 快速开始

```python
from HongMeng_raw_data_Parser import HongMengFileProcessor

processor = HongMengFileProcessor(verbose=True)
result = processor.process_file('data.dat', save=True)

# SPEC
spec_data = result['spec']['data']           # (n_fft, 4, 4096) int64
# VNA
s11 = result['vna']['data'][0]              # 第 0 次扫频 S11 复数
# TEMP
temp_raw = result['temp']['raw']             # (n_pkt,) bytes
```

```bash
python HongMeng_raw_data_Parser.py <file_path> [skip_pkt] [save]
```

---

## 相关模块

| 模块 | 说明 |
| ---- | ---- |
| `HongMeng_DataInspector.py` | `EffectiveDataExtractor` / `DataInspector`：按源分割、打印概览、绘制频谱/S11面板 |
| `HongMeng_VNA_Calibrator.py` | VNA OSL 校准引擎：误差项求解、修正、cable de-embedding |
| `dat_to_npz.py` | 命令行批量转换：`.dat` → `.npz` |

---

## result 数据结构

```
result
├── "spec"
│   ├── data              (n_fft, 4, 4096) int64   — 解码后科学数据
│   ├── raw               (n_pkt,) object           — 每包原始 bytes
│   ├── time/seq/src      (n_pkt,)                  — 时间戳/序列/源号
│   ├── obs_seq           (n_sources,) uint8        — 检测到的观测序列
│   └── metadata          dict                      — per-packet 包头字段
│       ├── app_id, version, pkt_type, sec_hdr_flag, group_flag
│       ├── data_len, valid_data_len, checksum
│
├── "vna"
│   ├── data              (n_sweep, n_freq) complex128 — S11 线性复数值
│   ├── raw               dict                      — 原始 IQ 值
│   │   ├── iref/qref     (n_sweep, n_freq) int64   — 入射 IQ
│   │   └── irfl/qrfl     (n_sweep, n_freq) int64   — 反射 IQ
│   ├── time/seq/src      (n_pkt,)                  — 同 SPEC
│   ├── obs_seq           (n_sources,) uint8        — 检测到的观测序列
│   ├── n_freq_per_sweep  (n_sweep,) int32           — 每次扫频频点数
│   └── metadata          dict                      — 同 SPEC + calc_total_count
│
└── "temp"
    ├── data              (n_pkt, 5, 5) float64     — 解码温度 (℃)，NaN=异常
    ├── raw               (n_pkt,) object           — 75B 原始数据
    ├── time/seq/src      (n_pkt,)
    └── metadata          dict
```

完整结构文档: [docs/result_structure.txt](docs/result_structure.txt)

---

## 输出文件

| 文件 | 说明 |
| ---- | ---- |
| `{stem}_Parced_v3.npz` | numpy 压缩归档（`save=True` 时生成） |
| `{stem}_parse.log`     | 解包日志（始终生成） |

---

## 参数

| 参数名             | 类型            | 默认值 | 说明                    |
| ------------------ | --------------- | ------ | ----------------------- |
| `file_path`        | `str` \| `Path` | 必需   | 输入文件路径            |
| `skip_pkt`         | `int`           | 0      | SPEC 跳过包数（对齐用） |
| `save`             | `bool`          | False  | 保存 NPZ               |
| `chunk_size`       | `int`           | 64MB   | 流式分块大小            |
| `stream_threshold` | `int`           | 512MB  | 流式读取阈值            |

---

## 版本历史

| 版本 | 日期 | 主要变更 |
| ---- | ---- | -------- |
| v4.0 | 2026-04-23 | 解包器全面性能优化、TEMP_PGA 修正(1→2)、VNA Calibrator、Streamlit 监视平台、dat_to_npz |
| v3.4 | 2026-04-08 | 不完整包处理、伪同步码防御、异常数据检测、DataInspector 独立模块 |
| v3.3 | 2026-04-07 | TEMP 温度解码（CVD）、obs_seq 观测序列还原 |
| v3.2 | 2026-04-03 | VNA S参数解码、解包日志、result 结构重构、多 bug 修复 |
| v3.0 | 2026-01-29 | 面向对象重构、流式读取、批量 numpy |
| v2.1 | 2026-01-06 | Checksum 校验 |
| v2.0 | 2025-12-16 | 多通道科学数据 |
| v1.0 | 2025-11-20 | 初始版本 |
