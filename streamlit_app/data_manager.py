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

from config import STREAM_THRESHOLD, CHUNK_SIZE


def _run_parser(file_path: str, file_name: str, file_size: int,
                progress_placeholder=None) -> tuple:
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
        chunk_size=CHUNK_SIZE,
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


def parse_uploaded_file(uploaded_file, progress_placeholder=None) -> tuple:
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
            progress_placeholder,
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


def parse_file_path(file_path: str, progress_placeholder=None) -> tuple:
    """Parse a .dat file from a local path."""
    p = Path(file_path)
    result, parser_info = _run_parser(
        file_path, p.name, p.stat().st_size,
        progress_placeholder,
    )
    return result, parser_info


def store_in_session(result: dict, parser_info: dict):
    """Store parsed data and diagnostics in session_state."""
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

    # Clear derived caches
    for k in list(st.session_state.keys()):
        if k.startswith('cache_'):
            del st.session_state[k]


def is_data_loaded() -> bool:
    return st.session_state.get('data_loaded', False)


def get_result() -> dict:
    return st.session_state.get('parsed_result', {})


def get_parser_info() -> dict:
    return st.session_state.get('parser_info', {})
