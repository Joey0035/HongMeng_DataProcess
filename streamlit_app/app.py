"""
app.py — Main entry point for the HongMeng monitoring platform.
"""

import streamlit as st
import traceback
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="HongMeng Monitor",
    layout="wide",
    page_icon="\U0001f52d",
)

import data_manager
from config import DEFAULT_VNA_FREQ, DEFAULT_TEMP_THRESHOLDS, ANOMALY_CATEGORIES
from theme import apply_theme

# ==============================================================
#  Theme
# ==============================================================

apply_theme()

# ==============================================================
#  Initialize session state defaults
# ==============================================================

if 'vna_freq_config' not in st.session_state:
    st.session_state['vna_freq_config'] = dict(DEFAULT_VNA_FREQ)
if 'temp_thresholds' not in st.session_state:
    st.session_state['temp_thresholds'] = dict(DEFAULT_TEMP_THRESHOLDS)
if 'parse_chunk_mb' not in st.session_state:
    st.session_state['parse_chunk_mb'] = 256

# ==============================================================
#  Sidebar
# ==============================================================

with st.sidebar:
    # Theme toggle
    is_light = st.toggle("Day Mode", value=(st.session_state['theme'] == 'light'), key="theme_toggle")
    new_theme = 'light' if is_light else 'dark'
    if new_theme != st.session_state['theme']:
        st.session_state['theme'] = new_theme
        st.rerun()

    st.divider()
    st.header("DATA LINK")

    # ── Advanced parser settings ─────────────────────────────────────
    with st.expander("Advanced Settings", expanded=True):
        chunk_mb = st.select_slider(
            "Streaming chunk size",
            options=[32, 64, 128, 256, 512, 1024],
            value=st.session_state['parse_chunk_mb'],
            format_func=lambda x: f"{x} MB",
            key="chunk_slider",
        )
        st.session_state['parse_chunk_mb'] = chunk_mb
        _chunk_bytes = chunk_mb * 1024 ** 2
        st.caption(
            f"Files ≥ 512 MB use streaming mode, reading in {chunk_mb} MB blocks. "
            "Larger = fewer I/O ops; smaller = less peak RAM."
        )

    # ── File path ────────────────────────────────────────────────────
    file_path = st.text_input(
        "File path (.dat)",
        placeholder="Paste path or drag file into Finder/Explorer to copy path",
    )
    if file_path:
        p = Path(file_path)
        if not p.exists():
            st.error("File not found.")
        else:
            size_bytes = p.stat().st_size
            size_gb    = size_bytes / 1024 ** 3
            st.caption(f"Size: {size_gb:.2f} GB")

            if st.button("PARSE", type="primary", use_container_width=True):
                progress_ph = st.empty()
                with st.spinner("Decoding telemetry..."):
                    try:
                        result, parser_info = data_manager.parse_file_path(
                            file_path, progress_placeholder=progress_ph,
                            chunk_size=_chunk_bytes)
                        data_manager.store_in_session(result, parser_info)
                        progress_ph.empty()
                        st.success(
                            f"LINKED // {parser_info['total_packets']} packets "
                            f"in {parser_info['parse_time_sec']:.1f}s"
                        )
                        st.session_state['parse_log'] = {
                            'success': True,
                            'parser_info': parser_info,
                        }
                    except Exception as e:
                        progress_ph.empty()
                        st.error(f"LINK FAILED: {e}")
                        st.session_state['parse_log'] = {
                            'success': False,
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                        }

    st.divider()

    # VNA frequency config
    if data_manager.is_data_loaded():
        result = data_manager.get_result()
        if 'vna' in result:
            st.subheader("VNA FREQ CONFIG")
            import numpy as np
            unique_nf = np.unique(result['vna']['n_freq_per_sweep'])
            vna_cfg = st.session_state['vna_freq_config']
            for nf in unique_nf:
                nf = int(nf)
                default_start, default_stop = vna_cfg.get(nf, (0, 100))
                col1, col2 = st.columns(2)
                with col1:
                    f0 = st.number_input(f"{nf}pts Start (MHz)", value=float(default_start),
                                         key=f"vna_f0_{nf}")
                with col2:
                    f1 = st.number_input(f"{nf}pts Stop (MHz)", value=float(default_stop),
                                         key=f"vna_f1_{nf}")
                vna_cfg[nf] = (f0, f1)
            st.session_state['vna_freq_config'] = vna_cfg

        st.divider()

        # Temperature thresholds
        st.subheader("TEMP THRESHOLDS")
        th = st.session_state['temp_thresholds']
        th['high'] = st.number_input("High Limit (°C)", value=th['high'], key="temp_th_high")
        th['low'] = st.number_input("Low Limit (°C)", value=th['low'], key="temp_th_low")
        st.session_state['temp_thresholds'] = th


# ==============================================================
#  Main area
# ==============================================================

st.title("HONGMENG HIGH-FREQ SA")

# ==============================================================
#  Parse output panel (shown after every parse attempt)
# ==============================================================

_pl = st.session_state.get('parse_log')
if _pl is not None:
    if not _pl['success']:
        # Parse failed — always expanded, full error highlighted
        with st.expander("PARSE OUTPUT", expanded=True):
            st.error(f"**LINK FAILED**\n\n```\n{_pl['error']}\n```")
            if _pl.get('traceback'):
                with st.expander("Full traceback"):
                    st.code(_pl['traceback'], language='python')
    else:
        _pi = _pl['parser_info']
        _n_dropped = len(_pi['dropped_records'])
        _n_errors  = _pi['error_count']
        _anomalies = _pi.get('anomalies', {})
        _n_anomaly = sum(
            len(v) if isinstance(v, (list, tuple)) else (len(v) if hasattr(v, '__len__') else 0)
            for v in _anomalies.values()
        )
        _has_issues = _n_errors > 0 or _n_dropped > 0 or _n_anomaly > 0

        with st.expander("PARSE OUTPUT", expanded=_has_issues):
            # ── File info ────────────────────────────────────────────
            st.caption(
                f"FILE: **{_pi['filename']}** // "
                f"SIZE: {_pi['file_size'] / 1024 / 1024:.1f} MB // "
                f"PARSE TIME: {_pi['parse_time_sec']:.1f}s // "
                f"TOTAL: {_pi['total_packets']} packets"
            )

            # ── Packet type counts ───────────────────────────────────
            _tc = _pi['type_counts']
            _stat_cols = st.columns(len(_tc) + 1)
            for _i, (_k, _v) in enumerate(_tc.items()):
                _stat_cols[_i].metric(_k, _v)
            _stat_cols[len(_tc)].metric("DROPPED", _n_dropped)

            # ── Errors ───────────────────────────────────────────────
            if _n_errors > 0:
                st.error(f"**{_n_errors} parse error(s)** encountered during decoding")

            # ── Dropped records ──────────────────────────────────────
            if _n_dropped > 0:
                st.warning(f"**{_n_dropped} record(s) dropped**")
                with st.expander(f"Dropped record details ({_n_dropped})"):
                    for _rec in _pi['dropped_records'][:50]:
                        st.text(str(_rec))
                    if _n_dropped > 50:
                        st.caption(f"... and {_n_dropped - 50} more")

            # ── Anomalies ────────────────────────────────────────────
            for _cat, _items in _anomalies.items():
                _n = len(_items) if hasattr(_items, '__len__') else 0
                if _n > 0:
                    _label = ANOMALY_CATEGORIES.get(_cat, _cat)
                    st.warning(f"**Anomaly [{_label}]:** {_n} instance(s)")

            if not _has_issues:
                st.success("No errors or anomalies detected.")

if not data_manager.is_data_loaded():
    _is_dark = st.session_state['theme'] == 'dark'
    _sub_color = '#4a5568' if _is_dark else '#718096'
    _hint_color = '#2d3748' if _is_dark else '#a0aec0'
    st.markdown(f"""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:4rem; margin-bottom:10px;">&#x1F52D;</div>
        <div style="font-size:1.1rem; color:{_sub_color}; letter-spacing:0.08em;">
            AWAITING DATA LINK
        </div>
        <div style="font-size:0.85rem; color:{_hint_color}; margin-top:20px; letter-spacing:0.05em;">
            Load a .dat file from the sidebar to initialize the monitoring system
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**PACKET STATUS**\n\nPacket counts, drop analysis, device health")
    with c2:
        st.markdown("**TEMPERATURE**\n\n25-point thermal monitoring with alerts")
    with c3:
        st.markdown("**SPECTRUM**\n\nSPEC visualization: 1D plots & waterfall")
    with c4:
        st.markdown("**S-PARAMETER**\n\nVNA S11: magnitude, phase, Smith chart")
else:
    parser_info = data_manager.get_parser_info()
    result = data_manager.get_result()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    tc = parser_info['type_counts']
    with col1:
        st.metric("SPEC", tc.get('SPEC', 0))
    with col2:
        st.metric("VNA", tc.get('VNA', 0))
    with col3:
        st.metric("TEMP", tc.get('TEMP', 0))
    with col4:
        st.metric("DROPPED", len(parser_info['dropped_records']),
                  delta=f"-{parser_info['error_count']} errors" if parser_info['error_count'] else None,
                  delta_color="inverse")

    # Time range
    time_range = st.session_state.get('time_range')
    if time_range:
        t0 = datetime.fromtimestamp(time_range[0])
        t1 = datetime.fromtimestamp(time_range[1])
        st.caption(
            f"FILE: **{parser_info['filename']}** // "
            f"SIZE: {parser_info['file_size'] / 1024 / 1024:.1f} MB // "
            f"TIME: {t0.strftime('%Y-%m-%d %H:%M:%S')} ~ {t1.strftime('%H:%M:%S')} // "
            f"PARSE: {parser_info['parse_time_sec']:.1f}s"
        )

    st.markdown("Navigate to subsystems via the sidebar.")
