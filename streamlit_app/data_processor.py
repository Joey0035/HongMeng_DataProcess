"""
data_processor.py — Pure computation layer for data splitting, statistics,
frequency axes, and time conversion. No Streamlit or Plotly imports.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from config import (
    SOURCE_LABEL_MAP, SPEC_BW_MHZ, SPEC_N_CH, PKTS_PER_FFT,
    TEMP_N_CHIPS, TEMP_CH_PER_CHIP, TEMP_SENSOR_LABELS,
    MAX_WATERFALL_ROWS,
)


# ==============================================================
#  Source name helper
# ==============================================================

def src_name(sid) -> str:
    return SOURCE_LABEL_MAP.get(int(sid), f'src_{sid}')


# ==============================================================
#  Frequency axes
# ==============================================================

def get_spec_freq_mhz() -> np.ndarray:
    return np.linspace(0, SPEC_BW_MHZ, SPEC_N_CH)


def get_vna_freq_mhz(n_freq: int, vna_freq_config: dict) -> Optional[np.ndarray]:
    if n_freq in vna_freq_config:
        f0, f1 = vna_freq_config[n_freq]
        return np.linspace(f0, f1, n_freq)
    return None


# ==============================================================
#  Per-FFT / Per-sweep helpers
# ==============================================================

def get_fft_src_time(spec: dict) -> tuple:
    """Returns (src_per_fft, time_per_fft) arrays."""
    n_fft = spec['data'].shape[0]
    step = PKTS_PER_FFT
    return spec['src'][::step][:n_fft], spec['time'][::step][:n_fft]


def get_vna_sweep_table(vna: dict) -> tuple:
    """Returns (src_per_sweep, n_freq_per_sweep) arrays."""
    gf = vna['metadata']['group_flag']
    sweep_starts = np.where(gf == 1)[0]
    n_sweep = vna['data'].shape[0]
    return vna['src'][sweep_starts[:n_sweep]], vna['n_freq_per_sweep'][:n_sweep]


def get_vna_sweep_times(vna: dict) -> np.ndarray:
    """Returns per-sweep timestamps (from the first packet of each sweep)."""
    gf = vna['metadata']['group_flag']
    sweep_starts = np.where(gf == 1)[0]
    n_sweep = vna['data'].shape[0]
    return vna['time'][sweep_starts[:n_sweep]]


# ==============================================================
#  Ordered source IDs (following obs_seq)
# ==============================================================

def _ordered_sids(sub: dict, unique_src: np.ndarray) -> list:
    obs = sub.get('obs_seq', np.array([]))
    ordered = list(obs) if len(obs) else sorted(unique_src)
    extra = [s for s in unique_src if int(s) not in [int(x) for x in ordered]]
    return list(ordered) + extra


# ==============================================================
#  Split by source
# ==============================================================

def split_spec_by_src(result: dict) -> dict:
    """Split SPEC data by source -> {src_name: (n_fft_src, 4, 4096)}"""
    spec = result['spec']
    data = spec['data']
    fft_src, fft_time = get_fft_src_time(spec)
    sids = _ordered_sids(spec, np.unique(fft_src))

    out = {}
    for sid in sids:
        mask = fft_src == int(sid)
        if mask.any():
            out[src_name(sid)] = data[mask]
    return out


def split_spec_by_src_with_time(result: dict) -> tuple:
    """Split SPEC data by source, also return per-FFT times.
    Returns ({src_name: data}, {src_name: time_array})
    """
    spec = result['spec']
    data = spec['data']
    fft_src, fft_time = get_fft_src_time(spec)
    sids = _ordered_sids(spec, np.unique(fft_src))

    data_out, time_out = {}, {}
    for sid in sids:
        mask = fft_src == int(sid)
        if mask.any():
            name = src_name(sid)
            data_out[name] = data[mask]
            time_out[name] = fft_time[mask]
    return data_out, time_out


def split_vna_by_src(result: dict, n_freq: Optional[int] = None) -> dict:
    """Split VNA data by source -> {src_name: (n_sweep_src, n_freq)}"""
    vna = result['vna']
    s11 = vna['data']
    sweep_src, sweep_nf = get_vna_sweep_table(vna)

    freq_mask = (sweep_nf == n_freq) if n_freq is not None else np.ones(len(sweep_src), dtype=bool)
    filtered_src = sweep_src[freq_mask]
    sids = _ordered_sids(vna, np.unique(filtered_src))

    out = {}
    for sid in sids:
        src_mask = (sweep_src == int(sid)) & freq_mask
        if not src_mask.any():
            continue
        name = src_name(sid)
        indices = np.where(src_mask)[0]
        if s11.dtype == object:
            sweeps = [s11[i] for i in indices]
            try:
                out[name] = np.stack(sweeps)
            except ValueError:
                out[name] = sweeps
        else:
            out[name] = s11[indices]
    return out


def split_vna_by_src_with_time(result: dict, n_freq: Optional[int] = None) -> tuple:
    """Split VNA data by source, also return per-sweep times."""
    vna = result['vna']
    s11 = vna['data']
    sweep_src, sweep_nf = get_vna_sweep_table(vna)
    sweep_times = get_vna_sweep_times(vna)

    freq_mask = (sweep_nf == n_freq) if n_freq is not None else np.ones(len(sweep_src), dtype=bool)
    filtered_src = sweep_src[freq_mask]
    sids = _ordered_sids(vna, np.unique(filtered_src))

    data_out, time_out = {}, {}
    for sid in sids:
        src_mask = (sweep_src == int(sid)) & freq_mask
        if not src_mask.any():
            continue
        name = src_name(sid)
        indices = np.where(src_mask)[0]
        time_out[name] = sweep_times[indices]
        if s11.dtype == object:
            sweeps = [s11[i] for i in indices]
            try:
                data_out[name] = np.stack(sweeps)
            except ValueError:
                data_out[name] = sweeps
        else:
            data_out[name] = s11[indices]
    return data_out, time_out


# ==============================================================
#  Time filtering
# ==============================================================

def filter_by_time_range(data: np.ndarray, time_arr: np.ndarray,
                         t_start: float, t_end: float) -> tuple:
    """Filter data by time range. Returns (filtered_data, filtered_time)."""
    mask = (time_arr >= t_start) & (time_arr <= t_end)
    return data[mask], time_arr[mask]


# ==============================================================
#  dB conversion
# ==============================================================

def to_dB_spec(arr: np.ndarray) -> np.ndarray:
    """SPEC: 10*log10(|x|)"""
    return 10 * np.log10(np.abs(arr).astype(float).clip(1e-30))


def to_dB_s11(arr: np.ndarray) -> np.ndarray:
    """S11: 20*log10(|x|)"""
    return 20 * np.log10(np.abs(arr).astype(float).clip(1e-30))


# ==============================================================
#  Time conversion
# ==============================================================

def timestamps_to_datetime(time_arr: np.ndarray) -> list:
    """Convert float64 UTC seconds to datetime objects."""
    return [datetime.fromtimestamp(float(t)) for t in time_arr]


def timestamps_to_datetime_strings(time_arr: np.ndarray, fmt: str = '%H:%M:%S') -> list:
    return [datetime.fromtimestamp(float(t)).strftime(fmt) for t in time_arr]


# ==============================================================
#  Temperature statistics
# ==============================================================

def compute_temp_statistics(temp_data: np.ndarray, time_arr: np.ndarray,
                            selected_indices: list, t_start: float, t_end: float) -> dict:
    """Compute temperature statistics for selected sensor points in a time range.

    Parameters
    ----------
    temp_data : (n_pkt, 5, 5) float64
    time_arr : (n_pkt,) float64
    selected_indices : list of (chip, ch) tuples
    t_start, t_end : time range

    Returns
    -------
    dict with 'global_max', 'global_min', 'per_point'
    """
    mask = (time_arr >= t_start) & (time_arr <= t_end)
    filtered = temp_data[mask]

    # Global stats across ALL points
    valid = filtered[~np.isnan(filtered)]
    global_max = float(np.nanmax(valid)) if len(valid) > 0 else float('nan')
    global_min = float(np.nanmin(valid)) if len(valid) > 0 else float('nan')

    per_point = {}
    for chip, ch in selected_indices:
        label = f"Chip{chip}-Ch{ch}"
        vals = filtered[:, chip, ch]
        valid_vals = vals[~np.isnan(vals)]
        if len(valid_vals) > 0:
            per_point[label] = {
                'max': float(np.max(valid_vals)),
                'min': float(np.min(valid_vals)),
                'mean': float(np.mean(valid_vals)),
                'fluctuation': float(np.max(valid_vals) - np.min(valid_vals)),
            }
        else:
            per_point[label] = {'max': float('nan'), 'min': float('nan'),
                                'mean': float('nan'), 'fluctuation': float('nan')}

    return {'global_max': global_max, 'global_min': global_min, 'per_point': per_point}


# ==============================================================
#  Sequence gap analysis
# ==============================================================

def compute_seq_gap_details(seq: np.ndarray, time_arr: np.ndarray, label: str) -> list:
    """Compute detailed sequence gap records for display.

    Returns list of dicts: {type, index, from_seq, to_seq, estimated_lost, timestamp}
    """
    if len(seq) < 2:
        return []

    actual_diff = np.diff(seq.astype(np.int32))
    actual_diff_wrapped = np.where(actual_diff == -16383, 1, actual_diff)
    gaps = np.where(actual_diff_wrapped != 1)[0]

    records = []
    for g in gaps:
        records.append({
            'type': label,
            'pkt_index': int(g),
            'from_seq': int(seq[g]),
            'to_seq': int(seq[g + 1]),
            'estimated_lost': abs(int(actual_diff_wrapped[g]) - 1),
            'timestamp': datetime.fromtimestamp(float(time_arr[g])).strftime('%Y-%m-%d %H:%M:%S'),
        })
    return records


# ==============================================================
#  Sensor index helpers
# ==============================================================

def label_to_chip_ch(label: str) -> tuple:
    """'Chip2-Ch3' -> (2, 3)"""
    parts = label.replace('Chip', '').replace('Ch', '').split('-')
    return int(parts[0]), int(parts[1])


def get_all_sensor_indices() -> list:
    """Return all (chip, ch) tuples in order."""
    return [(i, j) for i in range(TEMP_N_CHIPS) for j in range(TEMP_CH_PER_CHIP)]


# ==============================================================
#  Waterfall downsampling for large datasets
# ==============================================================

def downsample_waterfall(data_2d: np.ndarray, time_labels: list,
                         max_rows: int = None) -> tuple:
    """Downsample a 2D array (n_time, n_freq) for waterfall display.

    If n_time > max_rows, uniformly sample max_rows rows.
    Returns (data_2d_ds, time_labels_ds, was_downsampled).
    """
    if max_rows is None:
        max_rows = MAX_WATERFALL_ROWS
    n = data_2d.shape[0]
    if n <= max_rows:
        return data_2d, time_labels, False
    indices = np.linspace(0, n - 1, max_rows, dtype=int)
    return data_2d[indices], [time_labels[i] for i in indices], True
