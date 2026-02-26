# Changelog

>核心逻辑是按版本号和日期组织，并将改动分为固定的几个类别：Added (新增), Changed (修改), Deprecated (弃用), Removed (移除), Fixed (修复), Security (安全)。

## [Unreleased] (尚未发布的开发中功能)
### Added
- 针对天线阵列站点的温度监控 Streamlit 面板。
- 温度超过阈值时的 UI 红色报警提示。

### Changed
- 将接收机动态范围的计算基准调整为 28MHz 频段。

## [HongMeng_raw_data_Parser 1.0.0] - 2026-01-29
### Basic
- 完成第一个版本的解包程序开发，只支持**频谱数据**解包，包含以下特性：
1. 流式分块读取，支持GB级大文件（自动选择整读/分块模式）
2. 批量numpy操作，减少Python循环开销
3. 预分配内存，降低内存峰值30-50%
4. 结构化日志系统，实时进度追踪
5. 自动内存管理和垃圾回收
6. 完整错误捕获和跳过机制


## [HongMeng_raw_data_Parser 1.1.0] - 2026-2-26
### Added
- VNA数据解包
- 温度数据解包
### Fixed
- 
### Security
- Backup原来只有频谱数据解包的代码：./BACKUP/HongMeng_raw_data_Parser_onlySPEC.py