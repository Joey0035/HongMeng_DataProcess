# -*- coding: utf-8 -*-
"""
Author: JoeyXu
Date: 2026-04-07
Description: HongMeng原始数据解析
"""
# %%
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import struct
import sys
from typing import Optional, Tuple, Dict, List
import gc
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 常数定义
SYNC_WORD = b'\xEB\x90'  # 同步码
PACKET_HEADER_SIZE = 2 + 2 + 2 + 3   # 9 bytes：sync + pkt_id + seq_ctrl + data_len字段
PACKET_TIME_SIZE = 8                   # 时间码（副导头）
PACKET_SRC_NUM_SIZE = 1               # 被测源序列号
PACKET_VALID_LEN_SIZE = 3             # 有效数据域长度字段
PACKET_CHECKSUM_SIZE = 2              # 校验和
# 包数据域 = 时间(8) + 被测源序列号(1) + 有效数据域长度(3) + 有效数据(N) + 校验和(2)

PACKET_DATA_DOMAIN_OVERHEAD = PACKET_TIME_SIZE + PACKET_SRC_NUM_SIZE + PACKET_VALID_LEN_SIZE + PACKET_CHECKSUM_SIZE  # 14 bytes

# 应用过程标识符（11 bits）→ 数据类型
APP_ID_SPEC = 0x000   # 相关器数据，N=2304
APP_ID_VNA  = 0x7FF   # VNA数据，    N=2404
APP_ID_TEMP = 0x7C0   # 温度数据，   N=75
APP_ID_MAP  = {
    APP_ID_SPEC: 'SPEC',
    APP_ID_VNA:  'VNA',
    APP_ID_TEMP: 'TEMP',
}

# 各类型有效数据字节数（N）
SCI_DATA_SIZES = {
    'SPEC': 2304,
    'VNA':  2404,
    'TEMP': 75,
}

# 包数据域长度上限（用于过滤伪同步码导致的荒谬 data_len）
# 最大合法包 = VNA: OVERHEAD(14) + 2404 = 2418; 留 2x 余量
MAX_DATA_LEN = 8192

PACKETS_PER_SPEC = 64
CHANNELS_PER_SPEC = 4
VALUES_PER_CHANNEL = 4096
BYTES_PER_VALUE = 9

# VNA 常数
VNA_HEADER_SIZE = 4           # 计算请求总计数(2B) + 计算请求计数(2B)
VNA_IQ_BYTES = 6              # 每个 I/Q 值 48 bit = 6 字节
VNA_FREQ_POINT_SIZE = 24      # 每频点 = 4 × 6 字节 (Iref, Qref, Irfl, Qrfl)
VNA_MAX_FREQ_PER_PKT = 100   # 每包最多 100 个频点

# TEMP 常数
TEMP_N_CHIPS     = 5           # AD7124 芯片数
TEMP_CH_PER_CHIP = 5           # 每芯片通道数
TEMP_VREF        = 2.5         # 参考电压 (V)
TEMP_PGA         = 2           # 增益
TEMP_IIO         = 0.0005      # 激励电流 (A)
PT1000_A         = 3.9083e-3   # Callendar-Van Dusen 系数 A
PT1000_B         = -5.775e-7   # Callendar-Van Dusen 系数 B
PT1000_R0        = 1000.0      # PT1000 标称阻值 (Ω)

# 数据包类型字面量（供类型标注使用）
PktDataType = str  # 'SPEC' | 'VNA' | 'TEMP' | 'UNKNOWN'


@dataclass
class SpecPacket:
    sync: int
    version: int
    pkt_type: int
    sec_hdr_flag: int
    app_id: int           # 应用过程标识符（11 bits），区分 SPEC/VNA/TEMP
    group_flag: int
    seq_count: int
    data_len: int         # 包数据域字节数（含时间、src_num、valid_data_len、有效数据、校验和）
    seconds: int
    microseconds: int
    src_num: int          # 被测源序列号（8 bits）
    valid_data_len: int   # 有效数据字节数 N
    sci_data: bytes       # 有效数据（SPEC/VNA/TEMP 原始字节）
    checksum: int
    pkt_data_type: PktDataType  # 'SPEC' | 'VNA' | 'TEMP' | 'UNKNOWN'

    __slots__ = ('sync', 'version', 'pkt_type', 'sec_hdr_flag', 'app_id',
                 'group_flag', 'seq_count', 'data_len', 'seconds',
                 'microseconds', 'src_num', 'valid_data_len', 'sci_data',
                 'checksum', 'pkt_data_type')

    def __repr__(self) -> str:
        """十六进制格式显示关键字段"""
        return (f"SpecPacket(sync={self.sync:04x}, version={self.version:#b}"
                f"pkt_type={self.pkt_type:#b}, sec_hdr_flag={self.sec_hdr_flag:#b}, "
                f"app_id={self.app_id:04x}, group_flag={self.group_flag:#b}, "
                f"seq_count={self.seq_count}, data_len={self.data_len}, "
                f"time={self.seconds}.{self.microseconds:06d}, "
                f"src_num={self.src_num}, valid_data_len={self.valid_data_len}, "
                f"checksum={self.checksum:04x}), type={self.pkt_data_type}")


class PacketParser:

    def __init__(self, log_errors: bool = True):
        self.log_errors = log_errors
        self.error_count = 0
        self.packet_count = 0
        self.type_counts: Dict[str, int] = {}  # 各类型包计数
        # 丢弃包记录: [{reason, file_offset, raw_hex}]
        self.dropped_records: List[Dict] = []
        self._last_fail_reason: str = ""

    def calc_checksum(self, data: bytes) -> int:
        return int(np.frombuffer(data, dtype=np.uint8).sum(dtype=np.int64)) & 0xFFFF

    def parse_packet(self, buf: bytes | memoryview) -> Optional[SpecPacket]:
        """解析单个数据包

        包结构：
          sync(2) | pkt_id(2) | seq_ctrl(2) | data_len_field(3)  ← PACKET_HEADER_SIZE=9
          seconds(4) | microseconds(4)                            ← 副导头时间码
          src_num(1) | valid_data_len_field(3) | sci_data(N)      ← 有效数据域
          checksum(2)                                              ← 校验和（属于包数据域）
        其中 data_len_field 存储的值 = 包数据域字节数 - 1
             包数据域 = 8 + 1 + 3 + N + 2 = 14 + N
        总包长 = PACKET_HEADER_SIZE(9) + 包数据域(14 + N) = 23 + N
        """
        self._last_fail_reason = ""

        if len(buf) < PACKET_HEADER_SIZE + PACKET_DATA_DOMAIN_OVERHEAD:
            self._last_fail_reason = "packet_too_short"
            return None

        try:
            offset = 0

            # 1. 同步码
            sync = struct.unpack_from(">H", buf, offset)[0]
            if sync != 0xEB90:
                return None
            offset += 2

            # 2. 包标识
            pkt_id = struct.unpack_from(">H", buf, offset)[0]
            offset += 2
            version     = (pkt_id >> 13) & 0b111
            pkt_type    = (pkt_id >> 12) & 0b1
            sec_hdr_flag = (pkt_id >> 11) & 0b1
            app_id      = pkt_id & 0x7FF  # 应用过程标识符（11 bits）

            # 3. 包序控制
            seq_ctrl = struct.unpack_from(">H", buf, offset)[0]
            offset += 2
            group_flag = (seq_ctrl >> 14) & 0b11
            seq_count  = seq_ctrl & 0x3FFF

            # 4. 包数据域长度（24 bit，值 = 包数据域字节数 - 1）
            #    包数据域 = 时间(8) + src_num(1) + valid_data_len字段(3) + 科学数据(N) + 校验和(2) = 14 + N
            data_len_raw = int.from_bytes(buf[offset:offset + 3], "big")
            data_len = data_len_raw + 1  # 包数据域实际字节数
            offset += 3

            # 5. 副导头：时间码
            seconds, microseconds = struct.unpack_from(">II", buf, offset)
            offset += 8

            # 6. 有效数据域：被测源序列号（8 bits）
            src_num = buf[offset]
            offset += 1

            # 7. 有效数据域：有效数据域长度（24 bits，值 = N - 1）
            #    注意：valid_data_len 是逻辑有效字节数，可能小于物理分配空间
            #    物理分配空间 = data_len - time(8) - src_num(1) - valid_len_field(3) - checksum(2)
            valid_data_len_raw = int.from_bytes(buf[offset:offset + 3], "big")
            valid_data_len = valid_data_len_raw + 1  # 有效数据实际字节数 N
            offset += 3

            # 8. 有效数据域：科学/VNA/温度数据
            #    sci_data_space = 物理分配空间（含填充 0x7E）
            #    valid_data_len = 有效数据长度（不含填充）
            sci_data_space = data_len - PACKET_DATA_DOMAIN_OVERHEAD  # data_len - 14
            sci_data = bytes(buf[offset:offset + sci_data_space])
            offset += sci_data_space

            # 9. 校验和（覆盖副导头到有效数据末尾，含填充）
            checksum = struct.unpack_from(">H", buf, offset)[0]

            # 校验和验证：副导头(时间码) + 有效数据域(src_num + valid_data_len字段 + 全部科学数据含填充)
            checksum_start = PACKET_HEADER_SIZE  # offset 9
            checksum_end = offset                 # 9 + data_len - 2
            calc_sum = self.calc_checksum(buf[checksum_start:checksum_end])
            if calc_sum != checksum:
                self._last_fail_reason = (
                    f"checksum_mismatch(calc=0x{calc_sum:04X},stored=0x{checksum:04X})"
                )
                if self.log_errors:
                    self.error_count += 1
                    if self.error_count <= 10:
                        logger.debug(f"Checksum mismatch: calc=0x{calc_sum:04X}, "
                                    f"expected=0x{checksum:04X}")
                return None

            # 识别包数据类型
            pkt_data_type = APP_ID_MAP.get(app_id, 'UNKNOWN')

            # valid_data_len 与类型预期值不匹配时标记异常（仍保留此包）
            expected_n = SCI_DATA_SIZES.get(pkt_data_type)
            if expected_n is not None and valid_data_len != expected_n:
                # VNA 包的 valid_data_len 因频点数不同本来就不固定，不做此检查
                if pkt_data_type != 'VNA':
                    if self.log_errors and self.error_count <= 10:
                        logger.warning(
                            f"{pkt_data_type} valid_data_len={valid_data_len}, "
                            f"expected={expected_n} (seq={seq_count})")

            self.type_counts[pkt_data_type] = self.type_counts.get(pkt_data_type, 0) + 1
            self.packet_count += 1

            return SpecPacket(
                sync=sync,
                version=version,
                pkt_type=pkt_type,
                sec_hdr_flag=sec_hdr_flag,
                app_id=app_id,
                group_flag=group_flag,
                seq_count=seq_count,
                data_len=data_len,
                seconds=seconds,
                microseconds=microseconds,
                src_num=src_num,
                valid_data_len=valid_data_len,
                sci_data=sci_data,
                checksum=checksum,
                pkt_data_type=pkt_data_type,
            )
        except Exception as e:
            self._last_fail_reason = f"parse_exception({e})"
            if self.log_errors:
                self.error_count += 1
                if self.error_count <= 10:
                    logger.debug(f"Packet parse error: {e}")
            return None

    def separate_packets_by_type(
        self, packets: list
    ) -> Dict[str, list]:
        """
        Returns
        -------
        dict with keys 'SPEC', 'VNA', 'TEMP', 'UNKNOWN' (只含非空类型)
        """
        separated: Dict[str, list] = {}
        for pkt in packets:
            separated.setdefault(pkt.pkt_data_type, []).append(pkt)
        for dtype, lst in separated.items():
            logger.info(f"  {dtype:7s}: {len(lst):6d} packets")
        return separated

    def parse_packets_streaming(
        self,
        file_path: Path,
        chunk_size: int = 64 * 1024 * 1024,
        max_buffer: int = 256 * 1024 * 1024,
    ) -> list:
        """流式解析数据包"""
        packets = []
        buf = bytearray()          # bytearray.extend() 避免每次 += 全量拷贝
        file_size = file_path.stat().st_size
        bytes_read = 0
        buf_file_offset = 0  # buf[0] 对应的文件偏移

        logger.info(f"Starting streaming parse of {file_path.name} ({file_size / 1e9:.2f} GB)")

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                bytes_read += len(chunk)
                buf.extend(chunk)           # 原地追加，无拷贝

                # 处理缓冲区并获取未处理部分
                packets_chunk, buf, consumed = self._extract_packets_from_buffer(
                    buf, base_offset=buf_file_offset
                )
                buf_file_offset += consumed
                packets.extend(packets_chunk)

                # 动态缓冲区管理
                if len(buf) > max_buffer:
                    last_sync = buf.rfind(SYNC_WORD)
                    if last_sync > 0:
                        buf_file_offset += last_sync
                        buf = buf[last_sync:]
                    else:
                        buf_file_offset += len(buf) - 100
                        buf = buf[-100:]

                # 进度报告
                if bytes_read % (chunk_size * 10) == 0:
                    logger.info(f"Progress: {bytes_read / 1e9:.2f} GB / {file_size / 1e9:.2f} GB "
                               f"({len(packets)} packets)")

                # 内存管理
                if len(packets) % 30000 == 0:
                    gc.collect()

        # 处理剩余缓冲区
        if buf:
            packets_chunk, remaining, _ = self._extract_packets_from_buffer(
                buf, base_offset=buf_file_offset
            )
            packets.extend(packets_chunk)
            if remaining and remaining is not buf:
                self._log_trailing(remaining, base_offset=buf_file_offset + len(buf) - len(remaining))

        logger.info(f"Parsing complete: {len(packets)} packets, "
                   f"{self.error_count} errors")
        return packets

    def _extract_packets_from_buffer(
        self, buf: bytes | bytearray, base_offset: int = 0
    ) -> Tuple[list, bytes | bytearray, int]:
        """从缓冲区提取数据包

        Parameters
        ----------
        buf : bytes
        base_offset : int
            当前 buf 在原始文件中的起始偏移，用于记录丢弃包的文件位置

        Returns
        -------
        Tuple[list, bytes, int]
            (提取的packets列表, 未处理的buffer, 已消费的字节数)
        """
        packets = []
        mv = memoryview(buf)
        offset = 0
        buf_len = len(buf)
        first_sync_found = False

        while offset + PACKET_HEADER_SIZE <= buf_len:
            # 查找同步码 — 找不到时用 C 速度跳跃，避免逐字节 Python 循环
            if mv[offset:offset + 2] != SYNC_WORD:
                nxt = buf.find(SYNC_WORD, offset + 1)
                if nxt < 0:
                    offset = buf_len
                    break
                offset = nxt
                continue

            # 记录首次同步码前跳过的字节（不完整的头部数据）
            if not first_sync_found:
                first_sync_found = True
                if offset > 0:
                    logger.info(f"Skipped {offset} leading bytes before first sync at offset 0x{base_offset + offset:08X}")
                    self.dropped_records.append({
                        'reason':      'leading_incomplete_data',
                        'file_offset': base_offset,
                        'total_len':   offset,
                        'raw_hex':     bytes(mv[:min(offset, 64)]).hex() + ('...' if offset > 64 else ''),
                    })

            start = offset
            try:
                offset += 6
                if offset + 3 > buf_len:
                    offset = start
                    break

                data_len_raw = int.from_bytes(mv[offset:offset + 3], "big")
                data_len = data_len_raw + 1  # 包数据域字节数（含时间+src_num+valid_len+科学数据+校验和）
                offset += 3

                # 防御伪同步码：data_len 过大或过小→跳过此 sync，从 start+1 继续扫描
                if data_len < PACKET_DATA_DOMAIN_OVERHEAD or data_len > MAX_DATA_LEN:
                    offset = start + 1
                    continue

                # total_len = 主导头(9) + 包数据域(data_len)
                # data_len 已包含时间码、有效数据及校验和，无需再加 PACKET_CHECKSUM_SIZE
                total_len = PACKET_HEADER_SIZE + data_len
                if start + total_len > buf_len:
                    offset = start
                    break

                packet = self.parse_packet(mv[start:start + total_len])
                if packet:
                    packets.append(packet)
                else:
                    # 记录丢弃包
                    raw_bytes = bytes(mv[start:start + total_len])
                    self.dropped_records.append({
                        'reason':      self._last_fail_reason or 'unknown',
                        'file_offset': base_offset + start,
                        'total_len':   total_len,
                        'raw_hex':     raw_bytes.hex(),
                    })
                offset = start + total_len

            except Exception:
                offset = start + 1

        consumed = offset
        remaining_buf = buf[offset:] if offset > 0 else buf
        return packets, remaining_buf, consumed

    def _log_trailing(self, remaining: bytes, base_offset: int):
        """记录尾部不完整数据"""
        trail_len = len(remaining)
        if trail_len == 0:
            return
        logger.info(f"Trailing {trail_len} bytes at offset 0x{base_offset:08X} "
                     f"(incomplete packet at end, discarded)")
        self.dropped_records.append({
            'reason':      'trailing_incomplete_data',
            'file_offset': base_offset,
            'total_len':   trail_len,
            'raw_hex':     remaining[:min(trail_len, 64)].hex() + ('...' if trail_len > 64 else ''),
        })

    def parse_packets_all_in_memory(self, buf: bytes) -> list:
        """整读模式（小文件）"""
        packets, remaining, _ = self._extract_packets_from_buffer(buf, base_offset=0)
        if remaining and remaining is not buf:
            self._log_trailing(remaining, base_offset=len(buf) - len(remaining))
        return packets

# %%
class SciDataProcessor:

    @staticmethod
    def process_spec_block(packets: list) -> np.ndarray:
        """处理单个spec块（64个packets）— 单次 join + reshape，无逐通道循环"""
        if len(packets) != PACKETS_PER_SPEC:
            raise ValueError(f"Expected {PACKETS_PER_SPEC} packets, got {len(packets)}")
        # 一次性 join 全部 64 包（sci_data 已是 bytes，无需 bytes() 拷贝）
        all_bytes = b''.join(pkt.sci_data for pkt in packets)
        arr_u8 = np.frombuffer(all_bytes, dtype=np.uint8).reshape(
            CHANNELS_PER_SPEC, VALUES_PER_CHANNEL, BYTES_PER_VALUE
        )
        # Bytes 1–8 编码 int64（|value| < 2^63 对射电天文 FFT 输出成立）
        last8 = np.ascontiguousarray(arr_u8[:, :, 1:])   # (4, 4096, 8)
        return last8.view(np.dtype('>i8')).reshape(CHANNELS_PER_SPEC, VALUES_PER_CHANNEL)

    @staticmethod
    def _read_signed48(data: bytes, offset: int) -> int:
        """读取 48-bit 有符号整数 (big-endian, 二进制补码)"""
        val = int.from_bytes(data[offset:offset + 6], 'big')
        if val >= (1 << 47):
            val -= (1 << 48)
        return val

    @staticmethod
    def _decode_signed48_block(raw_bytes: bytes, n_values: int) -> np.ndarray:
        """Vectorized decode of n_values consecutive 6-byte big-endian signed48 values.

        Strategy: place each 6-byte chunk at the MSB of an 8-byte buffer (zero-fill
        the 2 LSB bytes), read as big-endian int64, arithmetic-right-shift 16 bits.
        This gives correct sign extension for any 48-bit two's-complement value.
        """
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_values, 6)
        padded = np.zeros((n_values, 8), dtype=np.uint8)
        padded[:, :6] = arr                                         # 6 bytes at MSB
        vals = padded.view(np.dtype('>i8')).reshape(n_values)       # big-endian int64
        return vals >> 16                                            # arithmetic right-shift

    @staticmethod
    def process_vna_sweep(sweep_pkts: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """处理单次 VNA 扫频（由 group_flag 分组的多个包）— numpy vectorized

        VNA 包科学数据域结构：
          [0:2]  计算请求总计数 (uint16)
          [2:4]  计算请求计数   (uint16)
          [4:]   频点数据，每频点 24 字节: Iref(6) Qref(6) Irfl(6) Qrfl(6)
                 最后一包有效频点数 = (valid_data_len - 4) / 24

        Returns
        -------
        (s11, iref, qref, irfl, qrfl) — 每个为 (n_freq,) complex128/int64
            s11:  S11 线性值 (复数), = (Irfl + j*Qrfl) / (Iref + j*Qref)
            iref, qref, irfl, qrfl: 原始 IQ 值 (int64)
        """
        freq_chunks = []
        total_freq = 0
        for pkt in sweep_pkts:
            sd = pkt.sci_data
            n_freq = (pkt.valid_data_len - VNA_HEADER_SIZE) // VNA_FREQ_POINT_SIZE
            if n_freq > 0:
                end = VNA_HEADER_SIZE + n_freq * VNA_FREQ_POINT_SIZE
                freq_chunks.append(bytes(sd[VNA_HEADER_SIZE:end]))
                total_freq += n_freq

        if total_freq == 0:
            empty = np.zeros(0, dtype=np.int64)
            return np.zeros(0, dtype=complex), empty, empty, empty, empty

        all_freq_bytes = b''.join(freq_chunks)
        # Reshape to (total_freq, 4 IQ components, 6 bytes each)
        raw = np.frombuffer(all_freq_bytes, dtype=np.uint8).reshape(total_freq, 4, 6)

        # Vectorized 48-bit signed decode: put 6 bytes at MSB of 8-byte buffer
        padded = np.zeros((total_freq, 4, 8), dtype=np.uint8)
        padded[:, :, :6] = raw
        vals = padded.view(np.dtype('>i8')).reshape(total_freq, 4) >> 16

        iref = vals[:, 0]
        qref = vals[:, 1]
        irfl = vals[:, 2]
        qrfl = vals[:, 3]

        # S11 = conj((Irfl + j*Qrfl) / (Iref + j*Qref))
        # 硬件 IQ 混频器输出约定为 I-jQ，取共轭修正相位符号
        incident   = iref.astype(np.float64) + 1j * qref.astype(np.float64)
        reflection = irfl.astype(np.float64) + 1j * qrfl.astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            s11 = np.where(np.abs(incident) > 0, np.conj(reflection / incident), 0.0 + 0j)

        return s11, iref, qref, irfl, qrfl

    @staticmethod
    def process_all_vna(vna_pkts: list) -> Dict[str, np.ndarray]:
        """处理所有 VNA 包，按 group_flag 分组成扫频

        Returns
        -------
        dict:
            's11':             (n_sweep, n_freq) complex128  — S11 线性值
            'iref':            (n_sweep, n_freq) int64
            'qref':            (n_sweep, n_freq) int64
            'irfl':            (n_sweep, n_freq) int64
            'qrfl':            (n_sweep, n_freq) int64
            'n_freq_per_sweep':(n_sweep,) int32   — 每次扫频的频点数
            'calc_total_count':(n_sweep,) uint16  — 计算请求总计数
        """
        if not vna_pkts:
            return {}

        # 按 group_flag 分组成扫频
        sweeps: List[list] = []
        current: list = []
        for pkt in vna_pkts:
            if pkt.group_flag == 1:  # 起始包
                if current:
                    sweeps.append(current)
                current = [pkt]
            else:
                current.append(pkt)
        if current:
            sweeps.append(current)

        logger.info(f"VNA: {len(sweeps)} sweeps from {len(vna_pkts)} packets")

        # 处理每次扫频
        sweep_s11, sweep_iref, sweep_qref, sweep_irfl, sweep_qrfl = [], [], [], [], []
        n_freq_list, total_count_list = [], []

        for sweep in sweeps:
            s11, iref, qref, irfl, qrfl = SciDataProcessor.process_vna_sweep(sweep)
            sweep_s11.append(s11)
            sweep_iref.append(iref)
            sweep_qref.append(qref)
            sweep_irfl.append(irfl)
            sweep_qrfl.append(qrfl)
            n_freq_list.append(len(s11))
            total_count_list.append(int.from_bytes(sweep[0].sci_data[0:2], 'big'))

        # 如果所有扫频频点数相同，可以堆成规整 2D 数组
        n_freq_arr = np.array(n_freq_list, dtype=np.int32)
        if np.all(n_freq_arr == n_freq_arr[0]):
            s11_out   = np.stack(sweep_s11)
            iref_out  = np.stack(sweep_iref)
            qref_out  = np.stack(sweep_qref)
            irfl_out  = np.stack(sweep_irfl)
            qrfl_out  = np.stack(sweep_qrfl)
        else:
            # 频点数不一致，用 object 数组
            s11_out   = np.empty(len(sweeps), dtype=object)
            iref_out  = np.empty(len(sweeps), dtype=object)
            qref_out  = np.empty(len(sweeps), dtype=object)
            irfl_out  = np.empty(len(sweeps), dtype=object)
            qrfl_out  = np.empty(len(sweeps), dtype=object)
            for i in range(len(sweeps)):
                s11_out[i]  = sweep_s11[i]
                iref_out[i] = sweep_iref[i]
                qref_out[i] = sweep_qref[i]
                irfl_out[i] = sweep_irfl[i]
                qrfl_out[i] = sweep_qrfl[i]

        return {
            's11':              s11_out,
            'iref':             iref_out,
            'qref':             qref_out,
            'irfl':             irfl_out,
            'qrfl':             qrfl_out,
            'n_freq_per_sweep': n_freq_arr,
            'calc_total_count': np.array(total_count_list, dtype=np.uint16),
        }

    @staticmethod
    def process_all_specs(packets: list) -> np.ndarray:
        """处理所有spec块"""
        n_specs = len(packets) // PACKETS_PER_SPEC
        result = np.zeros((n_specs, CHANNELS_PER_SPEC, VALUES_PER_CHANNEL), dtype=np.int64)

        for i in range(n_specs):
            spec_packets = packets[i * PACKETS_PER_SPEC:(i + 1) * PACKETS_PER_SPEC]
            result[i] = SciDataProcessor.process_spec_block(spec_packets)

            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} specs")

        return result

    @staticmethod
    def pt1000_resistance_to_temp(r: float) -> float:
        """PT1000 CVD 逆公式（Callendar-Van Dusen，适用 T ≥ 0℃）

        T = (-A + sqrt(A²-4B(1-R/R0))) / (2B)
        电阻超出 [800, 3000] Ω 认为传感器异常，返回 NaN。
        """
        if r < 800.0 or r > 3000.0:
            return np.nan
        discriminant = PT1000_A * PT1000_A - 4.0 * PT1000_B * (1.0 - r / PT1000_R0)
        if discriminant < 0.0:
            return np.nan
        return (-PT1000_A + np.sqrt(discriminant)) / (2.0 * PT1000_B)

    @staticmethod
    def process_all_temps(temp_pkts: list) -> np.ndarray:
        """解码温度包，返回 (n_pkt, 5, 5) float64 温度数组 (℃) — numpy vectorized

        75 bytes/packet = 5 chips × 5 channels × 3 bytes (offset-binary 24-bit)
        R  = (code − 2²³) × Vref / (2²³ × PGA × Iio)
        T  = PT1000 CVD 逆公式
        无效通道返回 NaN。
        """
        n = len(temp_pkts)
        sci_size = TEMP_N_CHIPS * TEMP_CH_PER_CHIP * 3  # 75 bytes
        n_sensors = TEMP_N_CHIPS * TEMP_CH_PER_CHIP      # 25

        scale       = TEMP_VREF / ((1 << 23) * TEMP_PGA * TEMP_IIO)
        offset_code = float(1 << 23)

        # Collect valid packets into a single flat buffer
        raw_chunks: list[bytes] = []
        valid_row = np.zeros(n, dtype=bool)
        for i, pkt in enumerate(temp_pkts):
            sd = pkt.sci_data
            if len(sd) >= sci_size:
                raw_chunks.append(bytes(sd[:sci_size]))
                valid_row[i] = True

        data = np.full((n, TEMP_N_CHIPS, TEMP_CH_PER_CHIP), np.nan, dtype=np.float64)
        if not raw_chunks:
            return data

        m = len(raw_chunks)
        # Stack into (m, n_sensors, 3) uint8, decode big-endian 24-bit unsigned
        raw_all = np.frombuffer(b''.join(raw_chunks), dtype=np.uint8).reshape(m, n_sensors, 3)
        codes = (raw_all[:, :, 0].astype(np.int64) << 16
                 | raw_all[:, :, 1].astype(np.int64) << 8
                 | raw_all[:, :, 2].astype(np.int64))

        R = (codes.astype(np.float64) - offset_code) * scale

        # Vectorized PT1000 CVD: NaN for out-of-range R or negative discriminant
        R_ok   = (R >= 800.0) & (R <= 3000.0)
        disc   = PT1000_A ** 2 - 4.0 * PT1000_B * (1.0 - R / PT1000_R0)
        disc_ok = disc >= 0.0
        ok     = R_ok & disc_ok

        T = np.where(ok,
                     (-PT1000_A + np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * PT1000_B),
                     np.nan)

        data[valid_row] = T.reshape(m, TEMP_N_CHIPS, TEMP_CH_PER_CHIP)
        return data


class MetadataExtractor:

    @staticmethod
    def validate_consistency(packets: list) -> Tuple[int, int, int]:
        """验证packet元数据一致性（首尾检查，不含 app_id，因其可能变化）"""
        if not packets:
            raise ValueError("No packets provided")

        first = packets[0]
        last = packets[-1]

        if first.version != last.version:
            raise ValueError(f"Version mismatch: {first.version} != {last.version}")
        if first.pkt_type != last.pkt_type:
            raise ValueError(f"Packet type mismatch: {first.pkt_type} != {last.pkt_type}")
        if first.sec_hdr_flag != last.sec_hdr_flag:
            raise ValueError("Secondary header flag mismatch")

        return first.version, first.pkt_type, first.sec_hdr_flag

    @staticmethod
    def extract_arrays(packets: list) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """提取主字段和 per-packet metadata，方便通过 metadata[field][i] 查找第 i 个包

        Returns
        -------
        (primary, metadata)
            primary:  {'time', 'seq', 'src'}         ← 常用主字段
            metadata: {'app_id', 'version', 'pkt_type', 'sec_hdr_flag',
                       'group_flag', 'data_len', 'valid_data_len', 'checksum'}
                       ← per-packet 包头字段，可按索引定位任意包
        """
        n = len(packets)

        # 用列表推导代替逐包索引赋值（更快，避免 n 次 Python 属性查找 + 数组写入）
        time_array     = np.array([pkt.seconds + pkt.microseconds * 1e-6 for pkt in packets], dtype=np.float64)
        seq_count      = np.array([pkt.seq_count      for pkt in packets], dtype=np.uint16)
        src_num_arr    = np.array([pkt.src_num        for pkt in packets], dtype=np.uint8)
        app_id_arr     = np.array([pkt.app_id         for pkt in packets], dtype=np.uint16)
        version_arr    = np.array([pkt.version        for pkt in packets], dtype=np.uint8)
        pkt_type_arr   = np.array([pkt.pkt_type       for pkt in packets], dtype=np.uint8)
        sec_hdr_arr    = np.array([pkt.sec_hdr_flag   for pkt in packets], dtype=np.uint8)
        group_flag_arr = np.array([pkt.group_flag     for pkt in packets], dtype=np.uint8)
        data_len_arr   = np.array([pkt.data_len       for pkt in packets], dtype=np.uint32)
        valid_len_arr  = np.array([pkt.valid_data_len for pkt in packets], dtype=np.uint32)
        checksum_arr   = np.array([pkt.checksum       for pkt in packets], dtype=np.uint16)

        primary = {
            'time': time_array,
            'seq':  seq_count,
            'src':  src_num_arr,
        }

        metadata = {
            'app_id':         app_id_arr,
            'version':        version_arr,
            'pkt_type':       pkt_type_arr,
            'sec_hdr_flag':   sec_hdr_arr,
            'group_flag':     group_flag_arr,
            'data_len':       data_len_arr,
            'valid_data_len': valid_len_arr,
            'checksum':       checksum_arr,
        }

        return primary, metadata


class HongMengFileProcessor:
    """主处理类"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.parser = PacketParser(log_errors=verbose)
        self.sci_processor = SciDataProcessor()
        self.metadata_extractor = MetadataExtractor()

    def process_file(
        self,
        file_path: str | Path,
        skip_pkt: int = 0,
        save: bool = False,
        chunk_size: int = 64 * 1024 * 1024,
        stream_threshold: int = 512 * 1024 * 1024,
    ) -> Dict:
        """处理原始数据文件

        Returns
        -------
        Dict，按数据类型组织，每种类型包含 data/raw/time/seq/src + metadata：

        result['spec'] = {
            'data':     np.ndarray (n_fft, 4, 4096),  # 解码后的科学数据
            'raw':      np.ndarray (n_pkt,) object,    # 每包原始科学数据 bytes
            'time':     np.ndarray (n_pkt,),           # 每包时间戳 float64
            'seq':      np.ndarray (n_pkt,),           # 包序列计数 uint16
            'src':      np.ndarray (n_pkt,),           # 被测源序列号 uint8
            'obs_seq':  np.ndarray (n_sources,),       # 检测到的观测序列 uint8
            'metadata': {                              # per-packet 包头字段
                'app_id':         np.ndarray uint16,   #   应用过程标识符
                'version':        np.ndarray uint8,    #   版本号
                'pkt_type':       np.ndarray uint8,    #   包类型
                'sec_hdr_flag':   np.ndarray uint8,    #   副导头标志
                'group_flag':     np.ndarray uint8,    #   分组标志
                'data_len':       np.ndarray uint32,   #   包数据域字节数
                'valid_data_len': np.ndarray uint32,   #   有效数据字节数
                'checksum':       np.ndarray uint16,   #   校验和
            },
        }

        result['vna']  = {
            'data':             (n_sweep, n_freq) complex128,  # S11 线性复数值
            'raw': {                                           # 原始 IQ 值
                'iref':         (n_sweep, n_freq) int64,
                'qref':         (n_sweep, n_freq) int64,
                'irfl':         (n_sweep, n_freq) int64,
                'qrfl':         (n_sweep, n_freq) int64,
            },
            'time':             (n_pkt,) float64,
            'seq':              (n_pkt,) uint16,
            'src':              (n_pkt,) uint8,
            'obs_seq':          (n_sources,) uint8,
            'n_freq_per_sweep': (n_sweep,) int32,
            'metadata': {
                ... per-packet 包头字段 (同 SPEC) ...,
                'calc_total_count': (n_sweep,) uint16,
            },
        }
        result['temp'] = {
            'data':     (n_pkt, 5, 5) float64,  # 解码温度 (℃)，NaN 表示无效通道
            'raw':      (n_pkt,) object,          # 每包原始 75-byte 科学数据
            'time':     (n_pkt,) float64,
            'seq':      (n_pkt,) uint16,
            'src':      (n_pkt,) uint8,
            'metadata': {...},
        }

        通过 metadata[field][i] 可快速定位第 i 个包的任意包头字段。
        """

        file_path = Path(file_path)
        logger.info(f"Starting DSL file processing: {file_path.name}")

        # 1. 解析数据包
        file_size = file_path.stat().st_size
        if file_size >= stream_threshold:
            logger.info(f"Using streaming mode (file size: {file_size / 1e9:.2f} GB)")
            packets = self.parser.parse_packets_streaming(file_path, chunk_size)
        else:
            logger.info(f"Using all-in-memory mode")
            with open(file_path, 'rb') as f:
                buf = f.read()
            packets = self.parser.parse_packets_all_in_memory(buf)

        if not packets:
            raise ValueError("No valid packets found")

        logger.info(f"Parsed {len(packets)} packets total")

        # 2. 按类型分离数据包
        logger.info("Separating packets by type:")
        separated = self.parser.separate_packets_by_type(packets)

        result: Dict = {}
        align_dropped: list = []  # SPEC 对齐丢弃的包

        # ---------- 2a. 相关器数据（SPEC）----------
        spec_pkts = separated.get('SPEC', [])
        if spec_pkts:
            aligned, align_dropped = self._align_packets(spec_pkts, skip_pkt)
            if aligned:
                self.metadata_extractor.validate_consistency(aligned)
                primary, metadata = self.metadata_extractor.extract_arrays(aligned)
                logger.info("Processing SPEC data...")
                spec_data = self.sci_processor.process_all_specs(aligned)
                logger.info(f"SPEC data shape: {spec_data.shape}")

                # per-packet 原始科学数据
                spec_raw = np.empty(len(aligned), dtype=object)
                for i, pkt in enumerate(aligned):
                    spec_raw[i] = pkt.sci_data

                result['spec'] = {
                    'data':     spec_data,
                    'raw':      spec_raw,
                    'time':     primary['time'],
                    'seq':      primary['seq'],
                    'src':      primary['src'],
                    'obs_seq':  self.detect_obs_sequence(primary['src']),
                    'metadata': metadata,
                }

        # ---------- 2b. VNA 数据 ----------
        vna_pkts = separated.get('VNA', [])
        if vna_pkts:
            primary, metadata = self.metadata_extractor.extract_arrays(vna_pkts)
            logger.info(f"VNA: {len(vna_pkts)} packets, processing S parameters...")

            # 解码 VNA S 参数
            vna_decoded = self.sci_processor.process_all_vna(vna_pkts)

            # sweep-level 字段移入 metadata
            metadata['calc_total_count'] = vna_decoded['calc_total_count']

            result['vna'] = {
                'data':             vna_decoded['s11'],          # (n_sweep, n_freq) complex128
                'raw': {                                         # 原始 IQ
                    'iref': vna_decoded['iref'],
                    'qref': vna_decoded['qref'],
                    'irfl': vna_decoded['irfl'],
                    'qrfl': vna_decoded['qrfl'],
                },
                'time':             primary['time'],             # (n_pkt,) float64
                'seq':              primary['seq'],              # (n_pkt,) uint16
                'src':              primary['src'],              # (n_pkt,) uint8
                'obs_seq':          self.detect_obs_sequence(primary['src']),  # (n_sources,) uint8
                'n_freq_per_sweep': vna_decoded['n_freq_per_sweep'],  # (n_sweep,) int32
                'metadata':         metadata,
            }

        # ---------- 2c. 温度数据（TEMP）----------
        temp_pkts = separated.get('TEMP', [])
        if temp_pkts:
            primary, metadata = self.metadata_extractor.extract_arrays(temp_pkts)
            logger.info(f"TEMP: {len(temp_pkts)} packets, decoding temperatures...")
            temp_raw = np.empty(len(temp_pkts), dtype=object)
            for i, pkt in enumerate(temp_pkts):
                temp_raw[i] = pkt.sci_data
            temp_data = self.sci_processor.process_all_temps(temp_pkts)
            logger.info(f"TEMP data shape: {temp_data.shape}")
            result['temp'] = {
                'data':     temp_data,
                'raw':      temp_raw,
                'time':     primary['time'],
                'seq':      primary['seq'],
                'src':      primary['src'],
                'metadata': metadata,
            }

        # ---------- 2d. 未知类型 ----------
        unk_pkts = separated.get('UNKNOWN', [])
        if unk_pkts:
            logger.warning(f"UNKNOWN type: {len(unk_pkts)} packets ignored")

        if save:
            self._save_result(file_path, result)

        # 异常检测
        anomalies = self._detect_anomalies(result)
        if anomalies:
            total_items = sum(len(v) for v in anomalies.values())
            logger.warning(f"Data anomalies detected: {total_items} issues in {len(anomalies)} categories")

        # 写解包日志（始终生成）
        log_path = (file_path.parent / f"{file_path.stem}_parse.log")
        self._write_parse_log(
            log_path=log_path,
            file_path=file_path,
            file_size=file_size,
            total_parsed=len(packets),
            separated=separated,
            spec_aligned_count=len(aligned) if spec_pkts else 0,
            align_dropped=align_dropped,
            unk_pkts=unk_pkts,
            anomalies=anomalies,
        )

        logger.info("Processing complete!")
        return result

    @staticmethod
    def _align_packets(packets: list, skip_pkt: int = 0) -> Tuple[list, list]:
        """对齐数据包，返回 (对齐后包列表, 被丢弃的包列表)"""
        usable = packets[skip_pkt:]
        fft_count = len(usable) // PACKETS_PER_SPEC
        keep_count = fft_count * PACKETS_PER_SPEC
        aligned = usable[:keep_count]
        dropped_by_skip = packets[:skip_pkt]
        dropped_by_tail = usable[keep_count:]
        dropped = dropped_by_skip + dropped_by_tail
        logger.info(f"Alignment: {fft_count} FFTs, {len(dropped)} SPEC packets dropped "
                    f"(skip={len(dropped_by_skip)}, tail={len(dropped_by_tail)})")
        return aligned, dropped

    @staticmethod
    def detect_obs_sequence(src: np.ndarray) -> np.ndarray:
        """从 src 数组中检测预设观测序列（源的排列组合）

        算法：RLE 压缩 src 获取连续源转换序列，找到首个重复出现的
        源值，截取两次出现之间的片段作为一个完整观测周期。

        Parameters
        ----------
        src : (n_pkt,) uint8   被测源序列号数组

        Returns
        -------
        obs_seq : (n_sources,) uint8   一个观测周期中源序列号（按观测顺序）
        """
        if len(src) == 0:
            return np.array([], dtype=np.uint8)

        # RLE: 连续相同值压缩，只保留转换点
        changes = np.where(np.diff(src) != 0)[0] + 1
        run_values = src[np.concatenate([[0], changes])]

        # 在 RLE 序列中找第一个重复出现的值
        seen: Dict[int, int] = {}
        for i, v in enumerate(run_values):
            v_int = int(v)
            if v_int in seen:
                return np.array(run_values[seen[v_int]:i], dtype=np.uint8)
            seen[v_int] = i

        # 未检测到完整周期，返回按出现顺序的去重序列
        _, idx = np.unique(run_values, return_index=True)
        return np.array(run_values[np.sort(idx)], dtype=np.uint8)

    @staticmethod
    def _detect_anomalies(result: Dict) -> Dict[str, List[str]]:
        """对已解包的数据执行异常检测

        检测项：
        1. 时间戳异常：时间=0、时间回跳（非单调递增）
        2. seq_count 间隙：相邻包序号不连续（考虑 14-bit 回绕）
        3. VNA 不完整扫频：首扫缺 group_flag==1 起始，尾扫频点数异常
        4. TEMP 全 NaN 行：某包全部通道无效

        Returns
        -------
        dict : {category: [description_strings]}
        """
        anomalies: Dict[str, List[str]] = {}

        def add(cat: str, msg: str):
            anomalies.setdefault(cat, []).append(msg)

        for key in ('spec', 'vna', 'temp'):
            if key not in result:
                continue
            sub = result[key]
            t = sub['time']
            seq = sub['seq']
            n = len(t)
            label = key.upper()

            # --- 时间戳异常 ---
            zero_mask = t == 0
            n_zero = int(zero_mask.sum())
            if n_zero > 0:
                add('timestamp', f"{label}: {n_zero} packets with timestamp=0")

            if n > 1:
                dt = np.diff(t)
                backwards = np.where(dt < 0)[0]
                if len(backwards) > 0:
                    add('timestamp',
                        f"{label}: {len(backwards)} time-backwards jumps "
                        f"(first at pkt #{int(backwards[0])}→#{int(backwards[0])+1}, "
                        f"Δt={dt[backwards[0]]:.6f}s)")

            # --- seq_count 间隙 ---
            if n > 1:
                expected_diff = np.ones(n - 1, dtype=np.int32)
                actual_diff = np.diff(seq.astype(np.int32))
                # 14-bit 回绕：0x3FFF → 0 的跳变 = -16383，等价于 +1
                actual_diff_wrapped = np.where(actual_diff == -16383, 1, actual_diff)
                gaps = np.where(actual_diff_wrapped != expected_diff)[0]
                if len(gaps) > 0:
                    total_missed = int(np.abs(actual_diff_wrapped[gaps] - 1).sum())
                    add('seq_gap',
                        f"{label}: {len(gaps)} seq_count discontinuities, "
                        f"~{total_missed} packets likely lost "
                        f"(first gap at pkt #{int(gaps[0])}: "
                        f"seq {int(seq[gaps[0]])}→{int(seq[gaps[0]+1])})")

        # --- VNA 不完整扫频 ---
        if 'vna' in result:
            vna = result['vna']
            nf = vna.get('n_freq_per_sweep')
            if nf is not None and len(nf) > 1:
                # 频点数分组（设备可能有多组配置，如 901 和 1901，属正常）
                vals, counts = np.unique(nf, return_counts=True)
                # 每组内的扫频应频点数一致；只标记出现次数=1的孤立扫频（很可能截断）
                singleton_sweeps = []
                for v, c in zip(vals, counts):
                    if c == 1:
                        idx = int(np.where(nf == v)[0][0])
                        singleton_sweeps.append((idx, int(v)))
                if singleton_sweeps:
                    details = ', '.join(f"sweep#{i}={n}pts" for i, n in singleton_sweeps[:5])
                    add('vna_sweep',
                        f"VNA: {len(singleton_sweeps)} singleton-frequency sweeps "
                        f"(likely truncated): {details}")
                # 报告频点分组概况
                if len(vals) > 1:
                    group_str = ', '.join(f"{int(v)}pts×{int(c)}" for v, c in zip(vals, counts))
                    add('vna_sweep', f"VNA freq groups: {group_str}")

            gf = vna['metadata']['group_flag']
            if len(gf) > 0 and gf[0] != 1:
                add('vna_sweep', "VNA: first packet missing group_flag=1 (incomplete leading sweep)")

        # --- TEMP 全 NaN ---
        if 'temp' in result:
            td = result['temp']['data']
            nan_rows = np.where(np.all(np.isnan(td.reshape(td.shape[0], -1)), axis=1))[0]
            if len(nan_rows) > 0:
                add('temp_nan',
                    f"TEMP: {len(nan_rows)} packets with all-NaN temperature "
                    f"(indices: {list(nan_rows[:10])}{'...' if len(nan_rows) > 10 else ''})")

        return anomalies

    @staticmethod
    def _save_result(file_path: Path, result: Dict):
        """保存结果（展平为 type_field / type_meta_field / type_raw_field 格式）"""
        output_path = file_path.parent / f"{file_path.stem}_Parced_v3.npz"
        logger.info(f"Saving to {output_path.name}")
        save_data = {}
        for type_key, sub_dict in result.items():
            if not isinstance(sub_dict, dict):
                continue
            for field, value in sub_dict.items():
                if field == 'metadata' and isinstance(value, dict):
                    for mf, mv in value.items():
                        if isinstance(mv, (np.ndarray, int, float, str)):
                            save_data[f"{type_key}_meta_{mf}"] = mv
                elif field == 'raw' and isinstance(value, dict):
                    # VNA raw dict: iref, qref, irfl, qrfl
                    for rf, rv in value.items():
                        if isinstance(rv, (np.ndarray, int, float, str)):
                            save_data[f"{type_key}_raw_{rf}"] = rv
                elif isinstance(value, (np.ndarray, int, float, str)):
                    save_data[f"{type_key}_{field}"] = value
        np.savez_compressed(output_path, **save_data)
        logger.info("Saved successfully")

    def _write_parse_log(
        self,
        log_path: Path,
        file_path: Path,
        file_size: int,
        total_parsed: int,
        separated: Dict[str, list],
        spec_aligned_count: int,
        align_dropped: list,
        unk_pkts: list,
        anomalies: Optional[Dict[str, List[str]]] = None,
    ):
        """写解包日志文件"""
        parser = self.parser
        lines: List[str] = []
        w = lines.append

        w(f"{'='*70}")
        w(f"  HongMeng Raw Data Parse Log")
        w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"{'='*70}")
        w("")

        # --- 1. 文件信息 ---
        w("[File Info]")
        w(f"  Path:       {file_path}")
        w(f"  Size:       {file_size:,} bytes ({file_size/1e6:.2f} MB)")
        w("")

        # --- 2. 解包总览 ---
        w("[Parse Summary]")
        w(f"  Total packets parsed OK:     {total_parsed}")
        w(f"  Parse errors (dropped):      {len(parser.dropped_records)}")
        w(f"  Total sync candidates:       {total_parsed + len(parser.dropped_records)}")
        w("")

        # --- 3. 各类型统计 ---
        w("[Packet Type Counts]")
        for dtype in ['SPEC', 'VNA', 'TEMP', 'UNKNOWN']:
            cnt = len(separated.get(dtype, []))
            if cnt > 0:
                w(f"  {dtype:7s}: {cnt:6d} packets")
        w("")

        # --- 4. SPEC 对齐统计 ---
        spec_total = len(separated.get('SPEC', []))
        if spec_total > 0:
            w("[SPEC Alignment]")
            w(f"  SPEC total parsed:    {spec_total}")
            w(f"  SPEC aligned (used):  {spec_aligned_count}  "
              f"({spec_aligned_count // PACKETS_PER_SPEC} FFTs x {PACKETS_PER_SPEC} pkts)")
            w(f"  SPEC alignment drop:  {len(align_dropped)}  "
              f"(tail packets not filling a complete {PACKETS_PER_SPEC}-packet FFT block)")
            w("")

        # --- 5. 丢弃包明细 ---
        all_drops: List[Dict] = []

        # 5a. 解析阶段丢弃（checksum / exception）
        for rec in parser.dropped_records:
            all_drops.append({
                'stage':       'parse',
                'reason':      rec['reason'],
                'file_offset': rec['file_offset'],
                'total_len':   rec['total_len'],
                'raw_hex':     rec['raw_hex'],
            })

        # 5b. SPEC 对齐丢弃
        for pkt in align_dropped:
            all_drops.append({
                'stage':       'spec_align',
                'reason':      'tail_not_filling_64pkt_block',
                'file_offset': None,
                'total_len':   PACKET_HEADER_SIZE + pkt.data_len,
                'raw_hex':     pkt.sci_data.hex(),
            })

        # 5c. UNKNOWN 类型
        for pkt in unk_pkts:
            all_drops.append({
                'stage':       'unknown_type',
                'reason':      f'unknown_app_id=0x{pkt.app_id:03X}',
                'file_offset': None,
                'total_len':   PACKET_HEADER_SIZE + pkt.data_len,
                'raw_hex':     pkt.sci_data.hex(),
            })

        w(f"[Dropped Packets Detail]  (total: {len(all_drops)})")
        if not all_drops:
            w("  (none)")
        else:
            w(f"  {'#':>4s}  {'Stage':<12s}  {'Offset':>12s}  {'PktLen':>7s}  Reason")
            w(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*7}  {'─'*40}")
            for idx, d in enumerate(all_drops):
                offset_str = f"0x{d['file_offset']:08X}" if d['file_offset'] is not None else "N/A"
                w(f"  {idx:4d}  {d['stage']:<12s}  {offset_str:>12s}  {d['total_len']:7d}  {d['reason']}")
        w("")

        # --- 6. 丢弃包原始 hex ---
        if all_drops:
            w(f"[Dropped Packets Raw Hex]")
            w("")
            for idx, d in enumerate(all_drops):
                offset_str = f"0x{d['file_offset']:08X}" if d['file_offset'] is not None else "N/A"
                w(f"--- #{idx}  stage={d['stage']}  offset={offset_str}  "
                  f"len={d['total_len']}  reason={d['reason']} ---")
                # 每行 64 个hex字符（32字节）
                hex_str = d['raw_hex']
                for j in range(0, len(hex_str), 64):
                    w(f"  {hex_str[j:j+64]}")
                w("")

        # --- 7. 数据异常检测结果 ---
        if anomalies is None:
            anomalies = {}
        total_anomaly_items = sum(len(v) for v in anomalies.values())
        w(f"[Data Anomalies]  (total: {total_anomaly_items} issues)")
        if not anomalies:
            w("  (none — all checks passed)")
        else:
            for cat, msgs in anomalies.items():
                w(f"  [{cat}]")
                for msg in msgs:
                    w(f"    - {msg}")
        w("")

        w(f"{'='*70}")
        w(f"  END OF LOG")
        w(f"{'='*70}")

        log_path.write_text('\n'.join(lines), encoding='utf-8')
        logger.info(f"Parse log saved to {log_path.name}")


# 向后兼容接口
def run_ParceSpecPacket(
    file_dir: str | Path,
    skip_pkt: int = 0,
    save: bool = False,
    chunk_size: int = 64 * 1024 * 1024,
    stream_threshold: int = 512 * 1024 * 1024,
) -> Dict:
    processor = HongMengFileProcessor(verbose=True)
    return processor.process_file(
        file_path=file_dir,
        skip_pkt=skip_pkt,
        save=save,
        chunk_size=chunk_size,
        stream_threshold=stream_threshold
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python HongMeng_raw_data_Parser.py <file_path> [skip_pkt] [save]")
        sys.exit(1)

    file_path = sys.argv[1]
    skip_pkt = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    save = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False

    result = run_ParceSpecPacket(file_path, skip_pkt=skip_pkt, save=save)
    print(f"Result keys: {list(result.keys())}")
    for type_key in result:
        sub = result[type_key]
        print(f"\n[{type_key}]:")
        for field, value in sub.items():
            if field == 'metadata':
                print(f"  metadata:")
                for mf, mv in value.items():
                    if hasattr(mv, 'shape'):
                        print(f"    {mf}: shape={mv.shape}, dtype={mv.dtype}")
                    else:
                        print(f"    {mf}: {mv}")
            elif field == 'raw' and isinstance(value, dict):
                print(f"  raw:")
                for rf, rv in value.items():
                    if hasattr(rv, 'shape'):
                        print(f"    {rf}: shape={rv.shape}, dtype={rv.dtype}")
                    else:
                        print(f"    {rf}: {rv}")
            elif hasattr(value, 'shape'):
                print(f"  {field}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"  {field}: list len={len(value)}")
            else:
                print(f"  {field}: {value}")

# %%
