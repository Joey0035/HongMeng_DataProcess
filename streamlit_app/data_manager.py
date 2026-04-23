"""
data_manager.py — Parser wrapper and session_state management.
This is the ONLY module that imports from the parent-directory parser.
"""

import sys
import time
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np

# Import parser from parent directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from HongMeng_raw_data_Parser import HongMengFileProcessor  # noqa: E402

import data_processor as dp
from config import STREAM_THRESHOLD, CHUNK_SIZE


def _run_parser(file_path: str, file_name: str, file_size: int,
                progress_placeholder=None, chunk_size: int = CHUNK_SIZE) -> tuple:
    """Shared parser logic for both upload and local path modes.

    Uses streaming mode automatically for large files (>= STREAM_THRESHOLD).
    """
    t0 = time.time()

    processor = HongMengFileProcessor(verbose=False)

    if progress_placeholder is not None:
        progress_placeholder.info(
            f"Parsing {file_name} ({file_size / 1024 / 1024:.1f} MB)..."
            + (" [streaming mode]" if file_size >= STREAM_THRESHOLD else "")
        )

    result = processor.process_file(
        file_path,
        chunk_size=chunk_size,
        stream_threshold=STREAM_THRESHOLD,
    )
    parse_time = time.time() - t0

    # Run anomaly detection (staticmethod, read-only on result)
    anomalies = HongMengFileProcessor._detect_anomalies(result)

    parser_info = {
        'filename': file_name,
        'file_size': file_size,
        'parse_time_sec': parse_time,
        'type_counts': dict(processor.parser.type_counts),
        'error_count': processor.parser.error_count,
        'dropped_records': list(processor.parser.dropped_records),
        'anomalies': anomalies,
        'total_packets': sum(processor.parser.type_counts.values()),
    }
    return result, parser_info


def parse_uploaded_file(uploaded_file, progress_placeholder=None,
                        chunk_size: int = CHUNK_SIZE) -> tuple:
    """Parse an uploaded .dat file.

    Writes uploaded data to a temp file in chunks to avoid doubling memory,
    then delegates to the parser (which uses streaming for large files).
    """
    # Write to temp file in chunks to avoid holding entire file twice in memory
    with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as tmp:
        tmp_path = tmp.name
        # Write in 8-MB chunks rather than getvalue() all at once
        chunk = uploaded_file.read(8 * 1024 * 1024)
        while chunk:
            tmp.write(chunk)
            chunk = uploaded_file.read(8 * 1024 * 1024)

    try:
        result, parser_info = _run_parser(
            tmp_path, uploaded_file.name, uploaded_file.size,
            progress_placeholder, chunk_size=chunk_size,
        )
    finally:
        # Clean up temp file and log
        try:
            Path(tmp_path).unlink()
            log_path = Path(tmp_path).with_name(Path(tmp_path).stem + '_parse.log')
            if log_path.exists():
                log_path.unlink()
        except OSError:
            pass

    return result, parser_info


def load_npz_file(npz_path: str, progress_placeholder=None) -> tuple:
    """Load a pre-processed .npz (produced by dat_to_npz.py) and reconstruct
    the result dict + parser_info that the rest of the app expects.
    """
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    t0 = time.time()
    if progress_placeholder is not None:
        progress_placeholder.info(f"Loading {p.name} ({p.stat().st_size / 1024**2:.1f} MB)...")

    npz = np.load(str(p), allow_pickle=True, mmap_mode='r')
    keys = set(npz.files)

    result = {}
    for type_key in ('spec', 'vna', 'temp'):
        type_keys = [k for k in keys if k.startswith(f'{type_key}_')]
        if not type_keys:
            continue

        sub: dict = {}
        metadata: dict = {}
        raw: dict = {}

        for k in sorted(type_keys):
            v = npz[k]
            # 0-d object arrays (scalars saved via allow_pickle) → unwrap
            if v.ndim == 0:
                v = v.item()

            if k.startswith(f'{type_key}_meta_'):
                field = k[len(f'{type_key}_meta_'):]
                metadata[field] = v
            elif k.startswith(f'{type_key}_raw_'):
                field = k[len(f'{type_key}_raw_'):]
                raw[field] = v
            else:
                field = k[len(f'{type_key}_'):]
                sub[field] = v

        if metadata:
            sub['metadata'] = metadata
        # VNA raw is a dict; SPEC/TEMP raw is an object array (may be absent if
        # the parser skipped object-dtype fields — that's fine, app doesn't need it)
        if raw:
            sub['raw'] = raw
        elif 'raw' not in sub:
            sub['raw'] = np.array([], dtype=object)

        result[type_key] = sub

    load_time = time.time() - t0

    # Reconstruct type_counts from the time arrays (one entry per packet)
    type_counts = {}
    for k in ('spec', 'vna', 'temp'):
        if k in result:
            time_arr = result[k].get('time', np.array([]))
            type_counts[k.upper()] = int(len(time_arr))

    parser_info = {
        'filename': p.name,
        'file_size': p.stat().st_size,
        'parse_time_sec': load_time,
        'type_counts': type_counts,
        'error_count': 0,
        'dropped_records': [],
        'anomalies': {},
        'total_packets': sum(type_counts.values()),
    }
    return result, parser_info


def parse_file_path(file_path: str, progress_placeholder=None,
                    chunk_size: int = CHUNK_SIZE) -> tuple:
    """Parse a .dat file from a local path."""
    p = Path(file_path)
    result, parser_info = _run_parser(
        file_path, p.name, p.stat().st_size,
        progress_placeholder, chunk_size=chunk_size,
    )
    return result, parser_info


def store_in_session(result: dict, parser_info: dict):
    """Store parsed data and diagnostics in session_state.
    Also pre-computes source splits so pages don't recompute on every widget interaction.
    """
    st.session_state['parsed_result'] = result
    st.session_state['parser_info'] = parser_info
    st.session_state['data_loaded'] = True

    # Compute time range across all types
    all_times = []
    for key in ('spec', 'vna', 'temp'):
        if key in result:
            t = result[key]['time']
            if len(t) > 0:
                all_times.append(t)
    if all_times:
        combined = np.concatenate(all_times)
        st.session_state['time_range'] = (float(combined.min()), float(combined.max()))

    # Pre-compute source splits — eliminates recomputation on every widget interaction
    try:
        if 'spec' in result:
            st.session_state['cache_spec_split'] = dp.split_spec_by_src_with_time(result)
        else:
            st.session_state.pop('cache_spec_split', None)

        if 'vna' in result:
            unique_nf = [int(x) for x in np.unique(result['vna']['n_freq_per_sweep'])]
            vna_cache = {}
            for nf in unique_nf:
                vna_cache[nf] = dp.split_vna_by_src_with_time(result, n_freq=nf)
            st.session_state['cache_vna_splits'] = vna_cache
        else:
            st.session_state.pop('cache_vna_splits', None)
    except Exception:
        pass  # splits computed on demand in pages if pre-computation fails

    # Clear derived caches
    for k in list(st.session_state.keys()):
        if k.startswith('cache_') and k not in ('cache_spec_split', 'cache_vna_splits'):
            del st.session_state[k]


def scan_npz_directory(dir_path: str) -> list:
    """Return metadata for every .npz file in *dir_path*, newest-first.

    Each entry: {name, path, size_mb, mtime_str}
    Returns an empty list when the directory is missing or contains no NPZ files.
    """
    d = Path(dir_path)
    if not d.is_dir():
        return []
    files = sorted(d.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        stat = p.stat()
        from datetime import datetime
        out.append({
            'name':     p.name,
            'path':     str(p),
            'size_mb':  stat.st_size / 1024 ** 2,
            'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        })
    return out


def is_data_loaded() -> bool:
    return st.session_state.get('data_loaded', False)


def get_result() -> dict:
    return st.session_state.get('parsed_result', {})


def get_parser_info() -> dict:
    return st.session_state.get('parser_info', {})
