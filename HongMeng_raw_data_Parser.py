# -*- coding: utf-8 -*-
"""
Author: JoeyXu
Date: 2026-01-29
Description: HongMeng原始数据解析
"""
# %%
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import struct
import sys
from typing import Optional, Tuple, Dict
import gc
import logging

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
# 注意：校验和属于包数据域的一部分，data_len 字段值已包含校验和
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

PACKETS_PER_SPEC = 64
CHANNELS_PER_SPEC = 4
VALUES_PER_CHANNEL = 4096
BYTES_PER_VALUE = 9

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

    def calc_checksum(self, data: bytes) -> int:
        return sum(data) & 0xFFFF

    def parse_packet(self, buf: bytes | memoryview) -> Optional[SpecPacket]:
        """解析单个数据包（新格式：支持 SPEC/VNA/TEMP 三种类型）

        包结构：
          sync(2) | pkt_id(2) | seq_ctrl(2) | data_len_field(3)  ← PACKET_HEADER_SIZE=9
          seconds(4) | microseconds(4)                            ← 副导头时间码
          src_num(1) | valid_data_len_field(3) | sci_data(N)      ← 有效数据域
          checksum(2)                                              ← 校验和（属于包数据域）
        其中 data_len_field 存储的值 = 包数据域字节数 - 1
             包数据域 = 8 + 1 + 3 + N + 2 = 14 + N
        总包长 = PACKET_HEADER_SIZE(9) + 包数据域(14 + N) = 23 + N
        """
        # 最小包长 = header(9) + 最小数据域(time+src_num+valid_len+min_data+checksum)
        # 最小数据域 = overhead(14) + 1字节数据 = 15，但实际不用卡这么紧
        # 只需确保能读完 header + 副导头 + src_num + valid_data_len 字段即可
        if len(buf) < PACKET_HEADER_SIZE + PACKET_DATA_DOMAIN_OVERHEAD:
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
            calc_sum = self.calc_checksum(bytes(buf[checksum_start:checksum_end]))
            if calc_sum != checksum:
                if self.log_errors:
                    self.error_count += 1
                    if self.error_count <= 10:
                        logger.debug(f"Checksum mismatch: calc=0x{calc_sum:04X}, "
                                    f"expected=0x{checksum:04X}")
                return None

            # 识别包数据类型
            pkt_data_type = APP_ID_MAP.get(app_id, 'UNKNOWN')
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
        buf = b""
        file_size = file_path.stat().st_size
        bytes_read = 0

        logger.info(f"Starting streaming parse of {file_path.name} ({file_size / 1e9:.2f} GB)")

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                bytes_read += len(chunk)
                buf += chunk

                # 处理缓冲区并获取未处理部分
                packets_chunk, buf = self._extract_packets_from_buffer(buf)
                packets.extend(packets_chunk)

                # 动态缓冲区管理
                if len(buf) > max_buffer:
                    # 保留最后可能包含sync的部分
                    last_sync = buf.rfind(SYNC_WORD)
                    if last_sync > 0:
                        buf = buf[last_sync:]
                    else:
                        buf = buf[-100:]  # 保留最后100字节

                # 进度报告
                if bytes_read % (chunk_size * 10) == 0:
                    logger.info(f"Progress: {bytes_read / 1e9:.2f} GB / {file_size / 1e9:.2f} GB "
                               f"({len(packets)} packets)")

                # 内存管理
                if len(packets) % 30000 == 0:
                    gc.collect()

        # 处理剩余缓冲区
        if buf:
            packets_chunk, _ = self._extract_packets_from_buffer(buf)
            packets.extend(packets_chunk)

        logger.info(f"Parsing complete: {len(packets)} packets, "
                   f"{self.error_count} errors")
        return packets

    def _extract_packets_from_buffer(self, buf: bytes) -> Tuple[list, bytes]:
        """从缓冲区提取数据包
        
        Returns
        -------
        Tuple[list, bytes]
            (提取的packets列表, 未处理的buffer)
        """
        packets = []
        mv = memoryview(buf)
        offset = 0
        buf_len = len(buf)

        while offset + PACKET_HEADER_SIZE <= buf_len:
            # 查找同步码
            if mv[offset:offset + 2] != SYNC_WORD:
                offset += 1
                continue

            start = offset
            try:
                offset += 6
                if offset + 3 > buf_len:
                    offset = start
                    break

                data_len_raw = int.from_bytes(mv[offset:offset + 3], "big")
                data_len = data_len_raw + 1  # 包数据域字节数（含时间+src_num+valid_len+科学数据+校验和）
                offset += 3

                # total_len = 主导头(9) + 包数据域(data_len)
                # data_len 已包含时间码、有效数据及校验和，无需再加 PACKET_CHECKSUM_SIZE
                total_len = PACKET_HEADER_SIZE + data_len
                if start + total_len > buf_len:
                    offset = start
                    break

                packet = self.parse_packet(mv[start:start + total_len])
                if packet:
                    packets.append(packet)
                offset = start + total_len

            except Exception:
                offset = start + 1

        # 返回已处理的packets和未处理的buffer
        remaining_buf = buf[offset:] if offset > 0 else buf
        return packets, remaining_buf

    def parse_packets_all_in_memory(self, buf: bytes) -> list:
        """整读模式（小文件）"""
        packets, _ = self._extract_packets_from_buffer(buf)
        return packets

# %%
class SciDataProcessor:
    """科学数据处理器 """

    @staticmethod
    def bytes_to_int64_vec(data_views: list) -> np.ndarray:
        """批量转换bytes到int64"""
        n = len(data_views)
        result = np.empty(n, dtype=np.int64)

        # SIMD优化的转换
        mask = (1 << 68) - 1
        offset = 1 << 68

        for i, data in enumerate(data_views):
            val = int.from_bytes(data, 'big') & mask
            result[i] = val - offset if (val >> 67) & 1 else val

        return result

    @staticmethod
    def process_spec_block(packets: list) -> np.ndarray:
        """处理单个spec块（64个packets）"""
        if len(packets) != PACKETS_PER_SPEC:
            raise ValueError(f"Expected {PACKETS_PER_SPEC} packets, got {len(packets)}")

        # 预分配结果数组
        result = np.zeros((CHANNELS_PER_SPEC, VALUES_PER_CHANNEL), dtype=np.int64)

        for class_idx in range(CHANNELS_PER_SPEC):
            start_pkt = class_idx * 16
            data_views = []

            # 批量收集字节块
            for pkt_idx in range(16):
                pkt = packets[start_pkt + pkt_idx]
                sci_data = pkt.sci_data
                for i in range(0, len(sci_data), BYTES_PER_VALUE):
                    if i + BYTES_PER_VALUE <= len(sci_data):
                        data_views.append(sci_data[i:i + BYTES_PER_VALUE])

            # 批量转换
            if data_views:
                result[class_idx] = SciDataProcessor.bytes_to_int64_vec(data_views)

        return result

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
                gc.collect()

        return result


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

        # 主字段
        time_array  = np.empty(n, dtype=np.float64)
        seq_count   = np.empty(n, dtype=np.uint16)
        src_num_arr = np.empty(n, dtype=np.uint8)

        # metadata — 包头完整字段，per-packet
        app_id_arr      = np.empty(n, dtype=np.uint16)
        version_arr     = np.empty(n, dtype=np.uint8)
        pkt_type_arr    = np.empty(n, dtype=np.uint8)
        sec_hdr_arr     = np.empty(n, dtype=np.uint8)
        group_flag_arr  = np.empty(n, dtype=np.uint8)
        data_len_arr    = np.empty(n, dtype=np.uint32)
        valid_len_arr   = np.empty(n, dtype=np.uint32)
        checksum_arr    = np.empty(n, dtype=np.uint16)

        for i, pkt in enumerate(packets):
            time_array[i]     = pkt.seconds + pkt.microseconds * 1e-6
            seq_count[i]      = pkt.seq_count
            src_num_arr[i]    = pkt.src_num

            app_id_arr[i]     = pkt.app_id
            version_arr[i]    = pkt.version
            pkt_type_arr[i]   = pkt.pkt_type
            sec_hdr_arr[i]    = pkt.sec_hdr_flag
            group_flag_arr[i] = pkt.group_flag
            data_len_arr[i]   = pkt.data_len
            valid_len_arr[i]  = pkt.valid_data_len
            checksum_arr[i]   = pkt.checksum

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

        result['vna']  = { 'raw', 'time', 'seq', 'src', 'metadata': {...} }
        result['temp'] = { 'raw', 'time', 'seq', 'src', 'metadata': {...} }

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

        # ---------- 2a. 相关器数据（SPEC）----------
        spec_pkts = separated.get('SPEC', [])
        if spec_pkts:
            aligned = self._align_packets(spec_pkts, skip_pkt)
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
                    'metadata': metadata,
                }

        # ---------- 2b. VNA 数据 ----------
        vna_pkts = separated.get('VNA', [])
        if vna_pkts:
            primary, metadata = self.metadata_extractor.extract_arrays(vna_pkts)
            logger.info(f"VNA: {len(vna_pkts)} packets")
            vna_raw = np.empty(len(vna_pkts), dtype=object)
            for i, pkt in enumerate(vna_pkts):
                vna_raw[i] = pkt.sci_data
            result['vna'] = {
                'raw':      vna_raw,
                'time':     primary['time'],
                'seq':      primary['seq'],
                'src':      primary['src'],
                'metadata': metadata,
            }

        # ---------- 2c. 温度数据（TEMP）----------
        temp_pkts = separated.get('TEMP', [])
        if temp_pkts:
            primary, metadata = self.metadata_extractor.extract_arrays(temp_pkts)
            logger.info(f"TEMP: {len(temp_pkts)} packets")
            temp_raw = np.empty(len(temp_pkts), dtype=object)
            for i, pkt in enumerate(temp_pkts):
                temp_raw[i] = pkt.sci_data
            result['temp'] = {
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

        logger.info("Processing complete!")
        return result

    @staticmethod
    def _align_packets(packets: list, skip_pkt: int = 0) -> list:
        """对齐数据包"""
        fft_count = len(packets[skip_pkt:]) // PACKETS_PER_SPEC
        aligned = packets[skip_pkt:fft_count * PACKETS_PER_SPEC + skip_pkt]
        dropped = len(packets) - len(aligned)
        logger.info(f"Alignment: {fft_count} FFTs, {dropped} packets dropped")
        return aligned

    @staticmethod
    def _save_result(file_path: Path, result: Dict):
        """保存结果（展平为 type_field / type_meta_field 格式）"""
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
                elif isinstance(value, (np.ndarray, int, float, str)):
                    save_data[f"{type_key}_{field}"] = value
        np.savez_compressed(output_path, **save_data)
        logger.info("Saved successfully")


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
            elif hasattr(value, 'shape'):
                print(f"  {field}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"  {field}: list len={len(value)}")
            else:
                print(f"  {field}: {value}")

# %%
