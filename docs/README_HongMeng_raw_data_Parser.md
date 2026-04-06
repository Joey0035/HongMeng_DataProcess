# HongMeng Raw Data Parser

## 概述

`HongMeng_raw_data_Parser.py` — DSL 高频频谱仪原始 `.dat` 数据解包与处理程序。

**版本**: v3.2 | **作者**: JoeyXu | **日期**: 2026-04-03

---

## 已实现功能

- SPEC 相关器数据解包（4路自相关/互相关，72-bit → int64）
- VNA S参数解码（入射/反射 IQ → S11 线性复数值）
- TEMP 温度数据解包（原始 bytes）
- Checksum 逐包校验
- 解包日志 `_parse.log`（丢弃包原因、位置、hex dump）
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
s11 = result['vna']['data']['s11'][0]        # 第 0 次扫频 S11 复数
# TEMP
temp_raw = result['temp']['raw']             # (n_pkt,) bytes
```

```bash
python HongMeng_raw_data_Parser.py <file_path> [skip_pkt] [save]
```

---

## result 数据结构

```
result
├── "spec"
│   ├── data              (n_fft, 4, 4096) int64   — 解码后科学数据
│   ├── raw               (n_pkt,) object           — 每包原始 bytes
│   ├── time/seq/src      (n_pkt,)                  — 时间戳/序列/源号
│   └── metadata          dict                      — per-packet 包头字段
│       ├── app_id, version, pkt_type, sec_hdr_flag, group_flag
│       ├── data_len, valid_data_len, checksum
│
├── "vna"
│   ├── data              dict                      — VNA 解码数据
│   │   ├── s11           (n_sweep, n_freq) complex128 — S11 线性值
│   │   ├── iref/qref     (n_sweep, n_freq) int64   — 入射 IQ
│   │   ├── irfl/qrfl     (n_sweep, n_freq) int64   — 反射 IQ
│   │   ├── sweep_time    (n_sweep,) float64         — 扫频起始时间
│   │   ├── sweep_src     (n_sweep,) uint8           — 扫频被测源
│   │   ├── n_freq_per_sweep (n_sweep,) int32        — 频点数
│   │   └── calc_total_count (n_sweep,) uint16       — 计算请求总计数
│   ├── raw/time/seq/src  (n_pkt,)                  — 同 SPEC
│   └── metadata          dict                      — 同 SPEC
│
└── "temp"
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
| v3.2 | 2026-04-03 | VNA S参数解码、解包日志、result 结构重构、多 bug 修复 |
| v3.0 | 2026-01-29 | 面向对象重构、流式读取、批量 numpy |
| v2.1 | 2026-01-06 | Checksum 校验 |
| v2.0 | 2025-12-16 | 多通道科学数据 |
| v1.0 | 2025-11-20 | 初始版本 |
