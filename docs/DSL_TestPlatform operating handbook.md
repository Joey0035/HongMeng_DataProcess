# DSL_TestPlatform Operating Handbook

## 概述

`HongMeng_raw_data_Parser.py` 用于解析 DSL FPGA 数据采集系统原始 `.dat` 文件，支持三种数据类型的解包与处理。

**版本**: v3.4
**作者**: JoeyXu
**日期**: 2026-04-08

---

## 主要特性

- **多类型数据解包**：SPEC 相关器 / VNA S参数 / 温度数据一次解析
- **VNA S参数解码**：自动按扫频分组，计算 S11 线性复数值
- **TEMP 温度解码**：offset-binary 24-bit → PT1000 CVD 逆公式 → ℃，输出 (n_pkt, 5, 5)
- **观测序列还原**：从 src 中自动检测预设观测序列（源的排列组合），输出 obs_seq
- **流式分块读取**：自动选择整读或分块模式（默认阈值 512MB）
- **解包日志**：每次解包自动生成 `_parse.log`，记录丢弃包详情和 hex 原始数据
- **不完整包自动剔除**：文件头 / 尾截断的不完整数据包自动识别、跳过并记录
- **伪同步码防御**：科学数据中出现 `0xEB90` 时通过 `data_len` 上限检查快速跳过
- **数据异常自动检测**：时间戳异常、seq_count 丢包、VNA 不完整扫频、TEMP 全 NaN 通道
- **Checksum 校验**：逐包验证，错误包自动丢弃并记录
- **ndarray 统一封装**：所有输出数据均为 numpy 数组

---

## 安装和依赖

- Python 3.9+
- numpy >= 1.20.0

```bash
pip install numpy
```

---

## 快速开始

### Python 调用

```python
from HongMeng_raw_data_Parser import HongMengFileProcessor

processor = HongMengFileProcessor(verbose=True)
result = processor.process_file('your_data.dat', save=True)

# SPEC 数据
spec = result['spec']
fft_data = spec['data']          # (n_fft, 4, 4096) int64
auto_1 = fft_data[0, 0, :]      # 第 0 次 FFT 的 Auto1

# VNA S参数
vna = result['vna']
s11 = vna['data'][0]              # 第 0 次扫频的 S11 复数
mag_dB = 20 * np.log10(np.abs(s11))
iref = vna['raw']['iref'][0]     # 第 0 次扫频入射 I

# 温度
temp = result['temp']
temp_data = temp['data']         # (n_pkt, 5, 5) float64, ℃，NaN=无效
t_pkt0 = temp_data[0]            # 第 0 包温度, shape=(5, 5)
temp_raw = temp['raw']           # 每包 75 字节原始数据
obs = result['spec']['obs_seq']  # 检测到的观测序列，如 [30,31,...,45,23]
```

### 兼容接口

```python
from HongMeng_raw_data_Parser import run_ParceSpecPacket

result = run_ParceSpecPacket('your_data.dat', skip_pkt=0, save=True)
```

### 命令行

```bash
python HongMeng_raw_data_Parser.py <file_path> [skip_pkt] [save]
python HongMeng_raw_data_Parser.py data.dat 0 true
```

---

## API 参数

### `HongMengFileProcessor.process_file()`

| 参数名             | 类型            | 默认值 | 说明                         |
| ------------------ | --------------- | ------ | ---------------------------- |
| `file_path`        | `str` \| `Path` | 必需   | 输入 `.dat` 文件路径         |
| `skip_pkt`         | `int`           | 0      | SPEC 跳过的包数量（用于对齐）|
| `save`             | `bool`          | False  | 是否保存为 `.npz` 文件       |
| `chunk_size`       | `int`           | 64MB   | 流式读取分块大小（字节）     |
| `stream_threshold` | `int`           | 512MB  | 触发流式读取的文件大小阈值   |

---

## 返回值结构

`process_file()` 返回 `dict`，按数据类型分为三个顶层 key：

```
result
├── "spec"   — 相关器数据 (app_id=0x000)
├── "vna"    — VNA S参数数据 (app_id=0x7FF)
└── "temp"   — 温度数据 (app_id=0x7C0)
```

### result["spec"]

```
spec
├── data              (n_fft, 4, 4096)  int64     解码后科学数据
│                      [i,0,:]=Auto1  [i,1,:]=Auto2
│                      [i,2,:]=Cross-Imag  [i,3,:]=Cross-Real
├── raw               (n_pkt,)          object    每包原始科学数据 bytes
├── time              (n_pkt,)          float64   时间戳
├── seq               (n_pkt,)          uint16    包序列计数
├── src               (n_pkt,)          uint8     被测源序列号
├── obs_seq           (n_sources,)      uint8     检测到的观测序列
└── metadata
    ├── app_id        (n_pkt,)          uint16
    ├── version       (n_pkt,)          uint8
    ├── pkt_type      (n_pkt,)          uint8
    ├── sec_hdr_flag  (n_pkt,)          uint8
    ├── group_flag    (n_pkt,)          uint8
    ├── data_len      (n_pkt,)          uint32
    ├── valid_data_len(n_pkt,)          uint32
    └── checksum      (n_pkt,)          uint16
```

> n_fft = n_pkt // 64。尾部不足 64 包的 SPEC 包被对齐丢弃（详见 parse.log）。

### result["vna"]

```
vna
├── data              (n_sweep, n_freq) complex128  S11 线性复数值
│                     或 (n_sweep,) object          = (Irfl+j*Qrfl)/(Iref+j*Qref)
├── raw               dict                         原始 IQ 值（按扫频组织）
│   ├── iref          (n_sweep, n_freq) int64       入射 I (48-bit signed)
│   ├── qref          (n_sweep, n_freq) int64       入射 Q
│   ├── irfl          (n_sweep, n_freq) int64       反射 I
│   └── qrfl          (n_sweep, n_freq) int64       反射 Q
├── time              (n_pkt,)          float64     每包时间戳
├── seq               (n_pkt,)          uint16      包序列计数
├── src               (n_pkt,)          uint8       被测源序列号
├── obs_seq           (n_sources,)      uint8       检测到的观测序列
├── n_freq_per_sweep  (n_sweep,)        int32       每次扫频频点数
└── metadata
    ├── app_id ~ checksum                           (同 SPEC, per-packet)
    └── calc_total_count (n_sweep,)     uint16      计算请求总计数
```

> S11 = (Irfl + j\*Qrfl) / (Iref + j\*Qref)。当各扫频频点数不一致时，data 和 raw 各字段退化为 `(n_sweep,) object` 数组。

### result["temp"]

```
temp
├── data              (n_pkt, 5, 5)    float64   解码温度 (℃)，NaN=异常/未接入
│                      [i, chip, ch]
├── raw               (n_pkt,)          object    每包原始科学数据 (75 bytes)
├── time              (n_pkt,)          float64   时间戳
├── seq               (n_pkt,)          uint16    包序列计数
├── src               (n_pkt,)          uint8     被测源序列号
└── metadata          ...                         (同 SPEC)
```

> 75 bytes = 5 chips × 5 channels × 3 bytes (offset-binary 24-bit ADC code)
> R = (code − 2²³) × 2.5 / (2²³ × 0.5mA)，T = PT1000 CVD 逆公式。

详细结构文档见 [result_structure.txt](result_structure.txt)

---

## 数据包格式

### 通用包结构

```
┌──────────────────────────────────────────────────────────────┐
│ 同步码 (2B): 0xEB90                                         │
│ 包标识 (2B): version(3b) | type(1b) | sec_hdr(1b) | app_id(11b) │
│ 包序控制 (2B): group_flag(2b) | seq_count(14b)              │
│ 数据域长度 (3B): 包数据域字节数 - 1                           │
├──── 包数据域 ────────────────────────────────────────────────┤
│ 时间码 (8B): seconds(4B) + microseconds(4B)                 │
│ 被测源序列号 (1B)                                            │
│ 有效数据域长度 (3B): 有效数据字节数 - 1                       │
│ 科学数据 (N B)                                               │
│ 校验和 (2B): 副导头+有效数据域 按字节累加取低 16bit          │
└──────────────────────────────────────────────────────────────┘
```

### 数据类型区分 (app_id)

| app_id | 类型 | 科学数据 N | 总包长 |
| ------ | ---- | ---------- | ------ |
| 0x000  | SPEC | 2304       | 2327   |
| 0x7FF  | VNA  | 2404       | 2427   |
| 0x7C0  | TEMP | 75         | 98     |

### VNA 包科学数据域

```
[0:2]  计算请求总计数 (uint16 big-endian)
[2:4]  计算请求计数   (uint16 big-endian)
[4:]   频点数据, 每频点 24 字节:
       Iref(6B) | Qref(6B) | Irfl(6B) | Qrfl(6B)
       各为 48-bit signed big-endian
       不足部分填充 0x7E
```

---

## 输出文件

| 文件 | 条件 | 说明 |
| ---- | ---- | ---- |
| `{stem}_Parced_v3.npz` | `save=True` | numpy 压缩归档 |
| `{stem}_parse.log`     | 始终生成     | 解包日志 |

### NPZ key 命名

```
{type}_{field}          — 主字段        如 spec_data, vna_data
{type}_meta_{field}     — metadata      如 spec_meta_app_id, vna_meta_calc_total_count
{type}_raw_{field}      — VNA raw 子dict  如 vna_raw_iref, vna_raw_qrfl
```

---

## 高级用法

```python
# 大文件：降低阈值启用流式
result = processor.process_file(
    'huge.dat',
    stream_threshold=256 * 1024 * 1024,
    chunk_size=32 * 1024 * 1024,
)

# SPEC 对齐：跳过前 N 个包
result = processor.process_file('data.dat', skip_pkt=64)
```

---

## 常见问题

### 内存不足
降低 `stream_threshold` 和 `chunk_size`。

### 解析错误过多
查看 `_parse.log` 中丢弃包详情。错误率 > 5% 建议检查原始数据。

### SPEC 数据量比包数少
SPEC 每 64 包组成一次 FFT，尾部不足 64 包的被丢弃。丢弃包记录在 parse.log 中。

### VNA S11 形状是 object 而非 2D
各扫频频点数不一致时自动退化为 object 数组，每个元素为 1D ndarray。

### 日志报告 "Data anomalies detected"
查看 `_parse.log` 末尾 `[Data Anomalies]` 段。常见异常含义：
- `timestamp`: 时间戳为 0 或时间回跳 → 可能是设备启动阶段或时钟重置
- `seq_gap`: seq_count 不连续 → 传输过程中有丢包
- `vna_sweep`: 不完整扫频或多种频点配置 → 文件头/尾截断或多配置混采
- `temp_nan`: 温度全 NaN → 传感器异常或未接入

### 文件头尾有不完整数据怎么办
无需处理，解析器自动跳过。跳过的字节数和位置记录在 `_parse.log` 的 `[Dropped Packets Detail]` 段中
（reason 为 `leading_incomplete_data` 或 `trailing_incomplete_data`）。

---

## 版本历史

### v3.4 (2026-04-08)
- 新增不完整包处理：文件头 / 尾截断数据自动跳过，位置记入 parse.log
- 新增伪同步码防御：`MAX_DATA_LEN` 上限检查，防止科学数据中的 `0xEB90` 误触发解析
- 新增 `valid_data_len` 校验：非 VNA 类型有效数据长度偏离标准值时发出 WARNING
- 新增 `_detect_anomalies()` 数据异常检测（时间戳/丢包/VNA扫频/TEMP全NaN）
- parse.log 新增 `[Data Anomalies]` 段
- `HongMeng_DataInspector.py`：DataInspector 从 notebook 中独立为 .py 模块
- 新增文档 `docs/异常数据处理说明.md`

### v3.3 (2026-04-07)
- 新增 TEMP 温度解码：offset-binary 24-bit → PT1000 CVD → (n_pkt, 5, 5) float64
- 新增 obs_seq 字段（SPEC / VNA）：RLE 检测 src 中的预设观测序列，还原源的排列组合

### v3.2 (2026-04-03)
- 新增 VNA S参数解码（S11 线性值、原始 IQ）
- 新增解包日志 `_parse.log`
- result 结构重构：metadata 子 dict、per-packet raw、ndarray 统一封装
- 修复 checksum 域长度计算 bug
- 修复 TEMP 包最小长度校验 bug
- 修复 VNA 包 sci_data 物理空间计算 bug

### v3.0 (2026-01-29)
- 完全重构，面向对象设计
- 流式分块读取、批量 numpy、预分配内存
- 结构化日志系统

### v2.1 (2026-01-06)
- 支持 checksum 校验

### v2.0 (2025-12-16)
- 重构数据解析逻辑，多通道科学数据

### v1.0 (2025-11-20)
- 初始版本

---

**最后更新**: 2026-04-08
