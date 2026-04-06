# DSL_TestPlatform Operating Handbook

## 概述

`HongMeng_raw_data_Parser.py` 用于解析 DSL FPGA 数据采集系统原始 `.dat` 文件，支持三种数据类型的解包与处理。

**版本**: v3.2
**作者**: JoeyXu
**日期**: 2026-04-03

---

## 主要特性

- **多类型数据解包**：SPEC 相关器 / VNA S参数 / 温度数据一次解析
- **VNA S参数解码**：自动按扫频分组，计算 S11 线性复数值
- **流式分块读取**：自动选择整读或分块模式（默认阈值 512MB）
- **解包日志**：每次解包自动生成 `_parse.log`，记录丢弃包详情和 hex 原始数据
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
auto_A = fft_data[0, 0, :]      # 第 0 次 FFT 的 A 路自相关

# VNA S参数
vna = result['vna']
s11 = vna['data']['s11'][0]      # 第 0 次扫频的 S11 复数
mag_dB = 20 * np.log10(np.abs(s11))

# 温度
temp = result['temp']
temp_raw = temp['raw']           # 每包 75 字节原始数据
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
│                      [i,0,:]=A自相关  [i,1,:]=B自相关
│                      [i,2,:]=互相关虚部  [i,3,:]=互相关实部
├── raw               (n_pkt,)          object    每包原始科学数据 bytes
├── time              (n_pkt,)          float64   时间戳
├── seq               (n_pkt,)          uint16    包序列计数
├── src               (n_pkt,)          uint8     被测源序列号
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
├── data                                          VNA 解码数据（按扫频组织）
│   ├── s11              (n_sweep, n_freq) complex128    S11 线性复数值
│   ├── iref             (n_sweep, n_freq) int64         入射 I (48-bit signed)
│   ├── qref             (n_sweep, n_freq) int64         入射 Q
│   ├── irfl             (n_sweep, n_freq) int64         反射 I
│   ├── qrfl             (n_sweep, n_freq) int64         反射 Q
│   ├── sweep_time       (n_sweep,)        float64       扫频起始时间
│   ├── sweep_src        (n_sweep,)        uint8         扫频被测源
│   ├── n_freq_per_sweep (n_sweep,)        int32         每次扫频频点数
│   └── calc_total_count (n_sweep,)        uint16        计算请求总计数
├── raw               (n_pkt,)            object    每包原始科学数据 bytes
├── time              (n_pkt,)            float64   时间戳
├── seq               (n_pkt,)            uint16    包序列计数
├── src               (n_pkt,)            uint8     被测源序列号
└── metadata          ...                           (同 SPEC)
```

> S11 = (Irfl + j\*Qrfl) / (Iref + j\*Qref)。当各扫频频点数不一致时，s11/iref/qref/irfl/qrfl 退化为 `(n_sweep,) object` 数组。

### result["temp"]

```
temp
├── raw               (n_pkt,)          object    每包原始科学数据 (75 bytes)
├── time              (n_pkt,)          float64   时间戳
├── seq               (n_pkt,)          uint16    包序列计数
├── src               (n_pkt,)          uint8     被测源序列号
└── metadata          ...                         (同 SPEC)
```

> 温度数据目前未做进一步解码。

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
{type}_{field}          — 主字段        如 spec_data, vna_raw
{type}_meta_{field}     — metadata      如 spec_meta_app_id
{type}_data_{field}     — VNA data 子dict  如 vna_data_s11
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

---

## 版本历史

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

**最后更新**: 2026-04-03
