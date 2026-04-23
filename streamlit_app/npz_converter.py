"""
npz_converter.py — Split a large .dat file into packet-aligned chunks and
convert each to a compressed .npz.

Bytes are processed directly in memory (no temp files), so each 1-GB chunk
requires only 1 source read + 1 NPZ write instead of the old 4-op sequence.
"""

import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SYNC_WORD = b'\xEB\x90'
DEFAULT_CHUNK_BYTES = 1 * 1024 ** 3  # 1 GB per NPZ chunk


# ==============================================================
#  Step 1 – packet-aligned split boundaries
# ==============================================================

def find_chunk_boundaries(file_path: Path, chunk_size: int = DEFAULT_CHUNK_BYTES) -> list:
    """Return byte offsets that divide *file_path* into ~*chunk_size* segments,
    each boundary snapped forward to the next ``\\xEB\\x90`` sync word.
    Always starts with 0, ends with file size.
    """
    file_size = file_path.stat().st_size
    boundaries = [0]
    with open(file_path, 'rb') as f:
        target = chunk_size
        while target < file_size:
            f.seek(target)
            lookahead = f.read(65536)
            idx = lookahead.find(SYNC_WORD)
            boundaries.append(target + idx if idx >= 0 else target)
            target += chunk_size
    boundaries.append(file_size)
    return boundaries


# ==============================================================
#  Step 2 – in-memory parse pipeline (no temp file)
# ==============================================================

def _process_chunk_bytes(buf: bytes, processor) -> tuple:
    """Run the full parse→process pipeline on raw bytes.

    Mirrors HongMengFileProcessor.process_file() but takes bytes directly
    instead of a file path, avoiding temp-file I/O entirely.

    Returns
    -------
    (result_dict, type_counts_dict)
    """
    from HongMeng_raw_data_Parser import HongMengFileProcessor  # static methods

    packets = processor.parser.parse_packets_all_in_memory(buf)
    if not packets:
        raise ValueError("No valid packets found in chunk")

    separated = processor.parser.separate_packets_by_type(packets)
    result: dict = {}
    align_dropped: list = []

    # ── SPEC ──────────────────────────────────────────────────────
    spec_pkts = separated.get('SPEC', [])
    if spec_pkts:
        aligned, align_dropped = HongMengFileProcessor._align_packets(spec_pkts)
        if aligned:
            processor.metadata_extractor.validate_consistency(aligned)
            primary, metadata = processor.metadata_extractor.extract_arrays(aligned)
            spec_data = processor.sci_processor.process_all_specs(aligned)
            spec_raw = np.empty(len(aligned), dtype=object)
            for i, pkt in enumerate(aligned):
                spec_raw[i] = pkt.sci_data
            result['spec'] = {
                'data':    spec_data,
                'raw':     spec_raw,
                'time':    primary['time'],
                'seq':     primary['seq'],
                'src':     primary['src'],
                'obs_seq': HongMengFileProcessor.detect_obs_sequence(primary['src']),
                'metadata': metadata,
            }

    # ── VNA ───────────────────────────────────────────────────────
    vna_pkts = separated.get('VNA', [])
    if vna_pkts:
        primary, metadata = processor.metadata_extractor.extract_arrays(vna_pkts)
        vna_decoded = processor.sci_processor.process_all_vna(vna_pkts)
        metadata['calc_total_count'] = vna_decoded['calc_total_count']
        result['vna'] = {
            'data': vna_decoded['s11'],
            'raw': {
                'iref': vna_decoded['iref'], 'qref': vna_decoded['qref'],
                'irfl': vna_decoded['irfl'], 'qrfl': vna_decoded['qrfl'],
            },
            'time':             primary['time'],
            'seq':              primary['seq'],
            'src':              primary['src'],
            'obs_seq':          HongMengFileProcessor.detect_obs_sequence(primary['src']),
            'n_freq_per_sweep': vna_decoded['n_freq_per_sweep'],
            'metadata':         metadata,
        }

    # ── TEMP ──────────────────────────────────────────────────────
    temp_pkts = separated.get('TEMP', [])
    if temp_pkts:
        primary, metadata = processor.metadata_extractor.extract_arrays(temp_pkts)
        temp_raw = np.empty(len(temp_pkts), dtype=object)
        for i, pkt in enumerate(temp_pkts):
            temp_raw[i] = pkt.sci_data
        temp_data = processor.sci_processor.process_all_temps(temp_pkts)
        result['temp'] = {
            'data':    temp_data,
            'raw':     temp_raw,
            'time':    primary['time'],
            'seq':     primary['seq'],
            'src':     primary['src'],
            'metadata': metadata,
        }

    return result, dict(processor.parser.type_counts)


# ==============================================================
#  Step 3 – NPZ serialisation
# ==============================================================

def _save_result_npz(result: dict, out_path: Path):
    """Save result dict to compressed NPZ (same key format as parser's _save_result)."""
    save_data: dict = {}
    for type_key, sub_dict in result.items():
        if not isinstance(sub_dict, dict):
            continue
        for field, value in sub_dict.items():
            if field == 'metadata' and isinstance(value, dict):
                for mf, mv in value.items():
                    if isinstance(mv, (np.ndarray, int, float, str)):
                        save_data[f"{type_key}_meta_{mf}"] = mv
            elif field == 'raw' and isinstance(value, dict):
                for rf, rv in value.items():
                    if isinstance(rv, (np.ndarray, int, float, str)):
                        save_data[f"{type_key}_raw_{rf}"] = rv
            elif isinstance(value, (np.ndarray, int, float, str)):
                save_data[f"{type_key}_{field}"] = value
    np.savez_compressed(str(out_path), **save_data)


# ==============================================================
#  Main entry point
# ==============================================================

def convert_dat_to_npz_chunks(
    file_path: str,
    out_dir: str,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> list:
    """Split *file_path* into packet-aligned ~1 GB segments and save each as NPZ.

    Each chunk is read once into memory, processed, then saved — no temp files.
    I/O per chunk: 1 source read + 1 NPZ write  (was: 4 ops with temp .dat).

    progress_cb(step, total_steps, message) where total_steps = n_chunks × 3.

    Returns list of dicts: {part, npz_path, type_counts, total_packets,
                             parse_time_sec, size_mb}
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from HongMeng_raw_data_Parser import HongMengFileProcessor

    src = Path(file_path).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    boundaries = find_chunk_boundaries(src, chunk_size)
    n_chunks   = len(boundaries) - 1
    total_steps = n_chunks * 3
    stem        = src.stem
    chunk_results = []

    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        raw_bytes = end - start
        npz_path  = out / f"{stem}_part{i + 1:02d}.npz"

        def _report(sub: int, msg: str):
            if progress_cb:
                progress_cb(i * 3 + sub, total_steps, msg)

        t0 = time.time()

        # 1 — read chunk from source (single sequential read)
        _report(0, f"[{i+1}/{n_chunks}] Reading {raw_bytes / 1024**2:.0f} MB …")
        with open(src, 'rb') as f:
            f.seek(start)
            buf = f.read(raw_bytes)

        # 2 — parse in memory (no temp file)
        _report(1, f"[{i+1}/{n_chunks}] Parsing …")
        processor = HongMengFileProcessor(verbose=False)
        chunk_result, type_counts = _process_chunk_bytes(buf, processor)
        del buf  # release before compressing

        # 3 — save NPZ
        _report(2, f"[{i+1}/{n_chunks}] Saving NPZ …")
        _save_result_npz(chunk_result, npz_path)

        chunk_results.append({
            'part':           i + 1,
            'npz_path':       str(npz_path),
            'type_counts':    type_counts,
            'total_packets':  sum(type_counts.values()),
            'parse_time_sec': time.time() - t0,
            'size_mb':        npz_path.stat().st_size / 1024 ** 2,
        })

    if progress_cb:
        progress_cb(total_steps, total_steps, "Conversion complete.")

    return chunk_results
