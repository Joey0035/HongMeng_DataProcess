"""
data_manager.py — Parser wrapper and session_state management.
This is the ONLY module that imports from the parent-directory parser.
"""

import sys
import time
import logging
import threading
import queue as _queue_mod
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

_PARSER_LOGGER = 'HongMeng_raw_data_Parser'


# ==============================================================
#  Log handlers
# ==============================================================

class _ListHandler(logging.Handler):
    """Capture log records into a list (synchronous use)."""
    def __init__(self):
        super().__init__()
        self.lines: list = []

    def emit(self, record):
        self.lines.append((record.levelname, self.format(record)))


class _QueueHandler(logging.Handler):
    """Push log records into a Queue (async/threaded use)."""
    def __init__(self, q: _queue_mod.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put((record.levelname, self.format(record)))


# ==============================================================
#  Core parser runner
# ==============================================================

def _run_parser(file_path: str, file_name: str, file_size: int,
                chunk_size: int = CHUNK_SIZE,
                log_handler: logging.Handler = None) -> tuple:
    """Run the parser with an optional log handler.

    If log_handler is None, a _ListHandler is created and its lines are
    stored in parser_info['parse_log_lines'].
    If a _QueueHandler is passed (async mode), parse_log_lines will be []
    — the caller accumulates lines from the queue independently.
    """
    t0 = time.time()
    processor = HongMengFileProcessor(verbose=True)

    _handler = log_handler if log_handler is not None else _ListHandler()
    _handler.setFormatter(logging.Formatter('%(message)s'))
    _plogger = logging.getLogger(_PARSER_LOGGER)
    _plogger.addHandler(_handler)
    try:
        result = processor.process_file(
            file_path,
            chunk_size=chunk_size,
            stream_threshold=STREAM_THRESHOLD,
        )
    finally:
        _plogger.removeHandler(_handler)

    parse_time = time.time() - t0
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
        # Populated only for sync _ListHandler; async caller fills this from queue
        'parse_log_lines': getattr(_handler, 'lines', []),
    }
    return result, parser_info


# ==============================================================
#  Async parse (background thread + queue)
# ==============================================================

def parse_file_path_async(file_path: str,
                          chunk_size: int = CHUNK_SIZE) -> tuple:
    """Start parsing a .dat file in a background thread.

    Returns (thread, log_queue, result_holder).
    - log_queue: yields (levelname, message) tuples; sentinel None when done
    - result_holder: dict populated on completion:
        success=True  → keys: result, parser_info
        success=False → keys: error, traceback
    """
    p = Path(file_path)
    log_queue: _queue_mod.Queue = _queue_mod.Queue()
    result_holder: dict = {}

    def _worker():
        q_handler = _QueueHandler(log_queue)
        try:
            result, parser_info = _run_parser(
                file_path, p.name, p.stat().st_size,
                chunk_size=chunk_size, log_handler=q_handler,
            )
            parser_info['parse_log_lines'] = []  # filled by caller from queue
            result_holder.update({'success': True, 'result': result,
                                   'parser_info': parser_info})
        except Exception as e:
            import traceback as _tb
            result_holder.update({'success': False, 'error': str(e),
                                   'traceback': _tb.format_exc()})
        finally:
            log_queue.put(None)  # sentinel: parsing is finished

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t, log_queue, result_holder


# ==============================================================
#  Sync helpers (kept for non-dat / small paths)
# ==============================================================

def parse_uploaded_file(uploaded_file, chunk_size: int = CHUNK_SIZE) -> tuple:
    """Parse an uploaded .dat file (synchronous)."""
    with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as tmp:
        tmp_path = tmp.name
        chunk = uploaded_file.read(8 * 1024 * 1024)
        while chunk:
            tmp.write(chunk)
            chunk = uploaded_file.read(8 * 1024 * 1024)
    try:
        result, parser_info = _run_parser(
            tmp_path, uploaded_file.name, uploaded_file.size,
            chunk_size=chunk_size,
        )
    finally:
        try:
            Path(tmp_path).unlink()
            log_path = Path(tmp_path).with_name(Path(tmp_path).stem + '_parse.log')
            if log_path.exists():
                log_path.unlink()
        except OSError:
            pass
    return result, parser_info


def parse_file_path(file_path: str, progress_placeholder=None,
                    chunk_size: int = CHUNK_SIZE) -> tuple:
    """Parse a .dat file from a local path (synchronous)."""
    p = Path(file_path)
    return _run_parser(file_path, p.name, p.stat().st_size, chunk_size=chunk_size)


def load_npz_file(npz_path: str, progress_placeholder=None) -> tuple:
    """Load a pre-processed .npz and reconstruct result + parser_info."""
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    t0 = time.time()
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
            if v.ndim == 0:
                v = v.item()
            if k.startswith(f'{type_key}_meta_'):
                metadata[k[len(f'{type_key}_meta_'):]] = v
            elif k.startswith(f'{type_key}_raw_'):
                raw[k[len(f'{type_key}_raw_'):]] = v
            else:
                sub[k[len(f'{type_key}_'):]] = v
        if metadata:
            sub['metadata'] = metadata
        if raw:
            sub['raw'] = raw
        elif 'raw' not in sub:
            sub['raw'] = np.array([], dtype=object)
        result[type_key] = sub

    load_time = time.time() - t0
    type_counts = {k.upper(): int(len(result[k].get('time', [])))
                   for k in ('spec', 'vna', 'temp') if k in result}
    parser_info = {
        'filename': p.name,
        'file_size': p.stat().st_size,
        'parse_time_sec': load_time,
        'type_counts': type_counts,
        'error_count': 0,
        'dropped_records': [],
        'anomalies': {},
        'total_packets': sum(type_counts.values()),
        'parse_log_lines': [(
            'INFO', f'Loaded {p.name} ({p.stat().st_size / 1024**2:.1f} MB) in {load_time:.2f}s'
        )],
    }
    return result, parser_info


# ==============================================================
#  Session state helpers
# ==============================================================

def store_in_session(result: dict, parser_info: dict):
    """Store parsed data in session_state and pre-compute source splits."""
    st.session_state['parsed_result'] = result
    st.session_state['parser_info'] = parser_info
    st.session_state['data_loaded'] = True

    all_times = []
    for key in ('spec', 'vna', 'temp'):
        if key in result:
            t = result[key]['time']
            if len(t) > 0:
                all_times.append(t)
    if all_times:
        combined = np.concatenate(all_times)
        st.session_state['time_range'] = (float(combined.min()), float(combined.max()))

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
        pass

    for k in list(st.session_state.keys()):
        if k.startswith('cache_') and k not in ('cache_spec_split', 'cache_vna_splits'):
            del st.session_state[k]


def scan_npz_directory(dir_path: str) -> list:
    """Return metadata for every .npz file in dir_path, newest-first."""
    d = Path(dir_path)
    if not d.is_dir():
        return []
    files = sorted(d.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        stat = p.stat()
        from datetime import datetime
        out.append({
            'name':      p.name,
            'path':      str(p),
            'size_mb':   stat.st_size / 1024 ** 2,
            'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        })
    return out


def is_data_loaded() -> bool:
    return st.session_state.get('data_loaded', False)


def get_result() -> dict:
    return st.session_state.get('parsed_result', {})


def get_parser_info() -> dict:
    return st.session_state.get('parser_info', {})
