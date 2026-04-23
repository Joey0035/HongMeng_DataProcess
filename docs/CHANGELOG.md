# Changelog

> 核心逻辑是按版本号和日期组织，并将改动分为固定的几个类别：Added (新增), Changed (修改), Deprecated (弃用), Removed (移除), Fixed (修复), Security (安全)。

## [Unreleased]

## [HongMeng DataProcess Tools 4.0.0] - 2026-04-23

### Added — 新模块

- **`HongMeng_VNA_Calibrator.py`**：VNA OSL（Open/Short/Load）1-port 3-term 误差模型校准引擎
  - `Keysight85033E`：Keysight 85033E 校准件标准模型，计算 Γ_open / Γ_short / Γ_load（含偏置延迟、fringing C / parasitic L 多项式模型）
  - `CableOffset`：RF 电缆 de-embedding 模型，支持解析传输线模型和 SNP 文件直接导入（`.s1p` / `.s2p`）
  - `solve_osl_error_terms()`：向量化 OSL 3-term 线性方程组求解（vectorized `np.linalg.solve`，含条件数预警）
  - `apply_osl_correction()`：1-port 误差修正，支持 `(n_freq,)` 和 `(n_sweep, n_freq)` 广播
  - `calibrate_vna_data()`：高层校准编排——逐周期独立定标、双校准平面（12-port 开关面 + LNA 面）、标准件自校核（quality check）
  - `read_touchstone()`：最小化 Touchstone `.s1p` / `.s2p` 解析器，支持 MA / RI / DB 格式

- **`streamlit_app/`**：完整 Streamlit 数据监视平台（`streamlit run app.py`）
  - **Page 1 — Packet Status**：SPEC / VNA / TEMP 包计数与占比、时间戳连续性图、seq_count 丢包统计
  - **Page 2 — Temperature**：5 × 5 传感器矩阵热图、时间序列曲线、超阈值红色告警、可配置告警阈值
  - **Page 3 — Spectrum**：SPEC 频谱按源分割，首次 / 均值折线图 + 瀑布图；支持时间范围过滤和通道选择（Auto1 / Auto2 / Cross-Real / Cross-Imag）
  - **Page 4 — S-Parameter**：VNA S11 按源分割，首次 / 均值折线图 + 瀑布图 + Smith 圆图；集成 OSL 校准，支持上传 SNP 文件做 cable de-embedding
  - 共享 `theme.py` CSS 主题，支持日间 / 夜间模式切换（sidebar toggle）
  - `data_manager.py`：全局数据缓存，支持直接上传 `.dat` 或加载 `.npz`，`mmap_mode='r'` 按需加载
  - `config.py`：可配置默认 VNA 频率范围、温度告警阈值、校准源名称

- **`dat_to_npz.py`**：命令行批量转换工具——将大 `.dat` 文件解包并保存为 `.npz`，适用于超过浏览器上传限制（>200 MB）的离线预处理场景

### Added — 解包器 (`HongMeng_raw_data_Parser.py`)

- `valid_data_len` 偏离预期值时输出 WARNING（VNA 不做检查，因频点数本来可变）
- 首包前跳过字节（leading incomplete data）记录到 `_parse.log` 和 `dropped_records`
- 尾部不完整数据（trailing incomplete packet）同样记录到日志

### Fixed

- **TEMP_PGA 值错误**：`TEMP_PGA` 从 `1` 修正为 `2`（对应 AD7124 PGA=2 设置），影响 PT1000 电阻计算结果及温度精度

### Changed — 性能优化 (2026-04-23)

详细说明见 [`docs/性能优化说明.md`](性能优化说明.md)。

| 位置 | 优化点 | 效果 |
|------|--------|------|
| `HongMeng_raw_data_Parser.py` | 流式缓冲从 `bytes +=` 改为 `bytearray.extend()` | 大文件内存分配 O(n²) → O(1) |
| 同上 | 同步字扫描改用 `buf.find()`（C 速 Boyer-Moore-Horspool） | 噪声包跳过速度提升 10×+ |
| 同上 | 校验和切片零拷贝（直接传 `memoryview` 切片） | 每包省去 ~2 KB 内存复制 |
| 同上 | SPEC 块解码：单次 `b''.join` + `reshape` + `.view('>i8')` | SPEC 块解码速度提升 3–5× |
| 同上 | 元数据提取改用列表推导式，移除热路径 `gc.collect()` | 消除预分配开销，减少 GC 延迟 |
| `streamlit_app/data_manager.py` | NPZ 以 `mmap_mode='r'` 加载 | 大文件初始加载时间和内存峰值显著降低 |
| `streamlit_app/data_processor.py` | 时间戳格式化改用 pandas 向量化 | 时间标签生成速度提升 20–100× |
| `streamlit_app/pages/2_Temperature.py` | `@st.cache_data` 缓存温度统计 | 参数不变时跳过重复统计计算 |
| `streamlit_app/pages/3_Spectrum.py` | `@st.cache_data` 缓存瀑布图预处理 | 切换通道 / 翻页时只重算变化信源 |
| `streamlit_app/plot_utils.py` | Smith 圆图改用 `go.Scattergl` + `customdata/hovertemplate` | 渲染速度提升 5–10×，hover 生成移至客户端 |
| 同上 | 瀑布图 `coloraxis` 配置批量 `update_layout` | N 次状态更新合并为 1 次 |
| `streamlit_app/pages/4_S_Parameter.py` | 预计算校准标准均值，SNP 临时文件 `try/finally` 清理 | 消除重复均值计算，防止文件泄漏 |

### Changed — 其他

- `theme.py` 从各页面内联样式提取为共享模块，修复全部页面日间模式显示异常

## [HongMeng_raw_data_Parser 3.4.0] - 2026-04-08
### Added
- 不完整包处理：文件头 / 尾截断数据自动识别并跳过，记录到 `_parse.log`
- 伪同步码防御：`MAX_DATA_LEN` 上限检查，防止 `0xEB90` 出现在科学数据中时误触发大内存分配
- `valid_data_len` 校验：SPEC/TEMP 有效数据长度偏离标准值时发出 WARNING
- 数据异常检测 `_detect_anomalies()`：
  - 时间戳异常（timestamp=0、时间回跳）
  - seq_count 间隙（丢包检测，含 14-bit 回绕处理）
  - VNA 不完整扫频（孤立频点数、首包缺 group_flag=1）
  - TEMP 全 NaN 行（整包无效通道）
- 解包日志新增 `[Data Anomalies]` 段，汇总所有异常检测结果
- `HongMeng_DataInspector.py`：从 notebook 中提取 `EffectiveDataExtractor` 和 `DataInspector` 类
  - `plot_vna_panel()` 支持 VNA S11 面板（first sweep / mean / waterfall，MHz x-axis，dB y-axis）
  - SPEC 和 VNA 频率轴统一使用 MHz，纵轴统一使用 dB
- 新增文档 `docs/异常数据处理说明.md`

### Changed
- DataInspector dtype 标签重命名：`'1'→Auto1, '2'→Auto2, 'r'→Cross-Real, 'i'→Cross-Imag`
- SPEC data 通道说明统一为：`[i,0,:]=Auto1, [i,1,:]=Auto2, [i,2,:]=Cross-Imag, [i,3,:]=Cross-Real`

## [HongMeng_raw_data_Parser 3.3.0] - 2026-04-07
### Added
- TEMP 温度解码：offset-binary 24-bit ADC code → PT1000 电阻 → CVD 逆公式 → ℃
  - 输出 `data` 字段，shape=(n_pkt, 5, 5) float64，chip×channel 组织，NaN 表示异常
- `obs_seq` 字段（SPEC / VNA）：RLE 检测 src 中的预设观测序列，还原源的排列组合

## [HongMeng_raw_data_Parser 3.2.0] - 2026-04-03
### Added
- VNA S参数解码：按 group_flag 自动分组扫频，计算 S11 线性复数值及原始 IQ
- 解包日志功能：每次解析自动生成 `_parse.log`，记录丢弃包原因、文件偏移、hex dump
- result 结构新增 `metadata` 子 dict，per-packet 包头字段集中管理
- SPEC `raw` 改为 per-packet（每包一个 bytes 条目）
- 所有输出数据统一 ndarray 封装
- app_id 改为 per-packet 数组（不再假定固定值）

### Fixed
- 修复 `total_len` 多加 2 字节 checksum 导致隔包丢失的 bug
- 修复最小长度校验误丢 TEMP 包（98 字节 < 100 字节阈值）
- 修复 `sci_data` 按 `valid_data_len` 而非物理空间切片导致 VNA checksum 错位

### Security
- Backup 原来只有频谱数据解包的代码：./BACKUP/HongMeng_raw_data_Parser_onlySPEC.py

## [HongMeng_raw_data_Parser 1.1.0] - 2026-02-26
### Added
- VNA 数据解包
- 温度数据解包

## [HongMeng_raw_data_Parser 1.0.0] - 2026-01-29
### Basic
- 完成第一个版本的解包程序开发，只支持**频谱数据**解包，包含以下特性：
1. 流式分块读取，支持GB级大文件（自动选择整读/分块模式）
2. 批量numpy操作，减少Python循环开销
3. 预分配内存，降低内存峰值30-50%
4. 结构化日志系统，实时进度追踪
5. 自动内存管理和垃圾回收
6. 完整错误捕获和跳过机制
