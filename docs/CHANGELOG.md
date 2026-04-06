# Changelog

> 核心逻辑是按版本号和日期组织，并将改动分为固定的几个类别：Added (新增), Changed (修改), Deprecated (弃用), Removed (移除), Fixed (修复), Security (安全)。

## [Unreleased] (尚未发布的开发中功能)
### Added
- 针对天线阵列站点的温度监控 Streamlit 面板。
- 温度超过阈值时的 UI 红色报警提示。

### Changed
- 将接收机动态范围的计算基准调整为 28MHz 频段。

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
