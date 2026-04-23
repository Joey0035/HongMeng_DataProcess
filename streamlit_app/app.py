"""
app.py — Main entry point for the HongMeng monitoring platform.

Layout:
  - Parse section (file input + PARSE button) lives in the MAIN area
  - Sidebar is minimal; only expands after data is loaded
  - Real-time log output during parsing via background thread + Queue polling
  - Packet Status shown inline on main page after loading (no separate page needed)
"""

import html as _html
import time
import traceback
import queue as _queue_mod
from queue import Empty as _QEmpty
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="HongMeng Monitor",
    layout="wide",
    page_icon="\U0001f52d",
)

import data_manager
import data_processor as dp
import plot_utils
from config import DEFAULT_VNA_FREQ, DEFAULT_TEMP_THRESHOLDS, ANOMALY_CATEGORIES
from theme import apply_theme, sidebar_colors, sidebar_label, sidebar_kv

apply_theme()

# ==============================================================
#  Session state defaults
# ==============================================================

_DEFAULTS = {
    'vna_freq_config':       dict(DEFAULT_VNA_FREQ),
    'temp_thresholds':       dict(DEFAULT_TEMP_THRESHOLDS),
    'parse_chunk_mb':        1024,
    'last_parsed_path':      None,
    'parse_log':             None,
    # global colormap (shared across all pages)
    'global_colormap_mag':   'Inferno',
    'global_colormap_phase': 'RdBu',
    # async parse state
    'parse_status':          'idle',
    'parse_log_buffer':      [],
    'parse_queue':           None,
    'parse_thread':          None,
    'parse_result_holder':   None,
    '_parsing_path':         '',
    '_parsing_filename':     '',
    '_parsing_size_mb':      0.0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ==============================================================
#  Helpers
# ==============================================================

def _render_parse_log(log_lines: list):
    if not log_lines:
        return
    level_class = {'INFO': 'll-INFO', 'WARNING': 'll-WARN',
                   'ERROR': 'll-ERROR', 'DEBUG': 'll-DEBUG'}
    rows = [
        f'<span class="{level_class.get(lv, "ll-INFO")}">{_html.escape(msg)}</span>'
        for lv, msg in log_lines
    ]
    st.markdown(
        f'<div class="parse-log-box">{"<br>".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def _drain_queue():
    q = st.session_state.get('parse_queue')
    if q is None:
        return True
    got_sentinel = False
    try:
        while True:
            item = q.get_nowait()
            if item is None:
                got_sentinel = True
                break
            st.session_state['parse_log_buffer'].append(item)
    except _QEmpty:
        pass
    return got_sentinel


def _finalize_parse():
    rh = st.session_state.get('parse_result_holder') or {}
    buf = list(st.session_state['parse_log_buffer'])
    if rh.get('success'):
        pi = rh['parser_info']
        pi['parse_log_lines'] = buf
        data_manager.store_in_session(rh['result'], pi)
        st.session_state['last_parsed_path'] = st.session_state['_parsing_path']
        st.session_state['parse_log'] = {
            'success': True,
            'parser_info': pi,
            'parse_log_lines': buf,
        }
    else:
        st.session_state['parse_log'] = {
            'success': False,
            'error': rh.get('error', 'Unknown error'),
            'traceback': rh.get('traceback', ''),
            'parse_log_lines': buf,
        }
    st.session_state['parse_status'] = 'idle'
    st.session_state['parse_queue'] = None
    st.session_state['parse_thread'] = None
    st.session_state['parse_result_holder'] = None


def _start_async_parse(file_path: str):
    _p = Path(file_path)
    _chunk = st.session_state['parse_chunk_mb'] * 1024 ** 2
    t, q, rh = data_manager.parse_file_path_async(str(_p), chunk_size=_chunk)
    st.session_state.update({
        'parse_status':        'running',
        'parse_log_buffer':    [],
        'parse_queue':         q,
        'parse_thread':        t,
        'parse_result_holder': rh,
        '_parsing_path':       str(_p),
        '_parsing_filename':   _p.name,
        '_parsing_size_mb':    _p.stat().st_size / 1024 ** 2,
    })


def _start_sync_npz(file_path: str):
    _p = Path(file_path)
    try:
        result, parser_info = data_manager.load_npz_file(str(_p))
        data_manager.store_in_session(result, parser_info)
        st.session_state['last_parsed_path'] = str(_p)
        st.session_state['parse_log'] = {
            'success': True,
            'parser_info': parser_info,
            'parse_log_lines': parser_info.get('parse_log_lines', []),
        }
    except Exception as e:
        st.session_state['parse_log'] = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'parse_log_lines': [],
        }


# ==============================================================
#  Polling loop — runs at top of every script cycle when active
# ==============================================================

if st.session_state.get('parse_status') == 'running':
    t_thread = st.session_state.get('parse_thread')
    got_sentinel = _drain_queue()

    with st.sidebar:
        is_light = st.toggle("Day Mode",
                             value=(st.session_state['theme'] == 'light'),
                             key="theme_toggle")
        if (('light' if is_light else 'dark') != st.session_state['theme']):
            st.session_state['theme'] = 'light' if is_light else 'dark'
            st.rerun()
        _acc_p = '#00f0ff' if st.session_state['theme'] == 'dark' else '#1a365d'
        _dim_p = '#8899aa' if st.session_state['theme'] == 'dark' else '#4a5568'
        st.divider()
        st.markdown(
            f'<div style="color:{_acc_p}; font-size:0.68rem; letter-spacing:0.12em;'
            f' font-weight:700; text-transform:uppercase; margin-bottom:6px;">PARSING</div>'
            f'<div style="color:{_dim_p}; font-size:0.78rem;">'
            f'{st.session_state.get("_parsing_filename", "")}<br>'
            f'{st.session_state.get("_parsing_size_mb", 0):.0f} MB</div>',
            unsafe_allow_html=True,
        )

    st.title("HONGMENG HIGH-FREQ SA")
    fname = st.session_state.get('_parsing_filename', '')
    fsize_mb = st.session_state.get('_parsing_size_mb', 0)
    st.markdown(f"**Parsing** `{fname}` ({fsize_mb:.0f} MB) …")

    buf = st.session_state['parse_log_buffer']
    if buf:
        _render_parse_log(buf)
    else:
        st.info("Initializing parser …")

    thread_alive = t_thread is not None and t_thread.is_alive()
    if got_sentinel or not thread_alive:
        try:
            while True:
                item = st.session_state['parse_queue'].get_nowait()
                if item is None:
                    break
                st.session_state['parse_log_buffer'].append(item)
        except (_QEmpty, AttributeError):
            pass
        _finalize_parse()
        st.rerun()
    else:
        time.sleep(0.1)   # faster polling → ~10 updates/sec
        st.rerun()

    st.stop()


# ==============================================================
#  Sidebar — idle state
# ==============================================================

with st.sidebar:
    is_light = st.toggle("Day Mode",
                         value=(st.session_state['theme'] == 'light'),
                         key="theme_toggle")
    if (('light' if is_light else 'dark') != st.session_state['theme']):
        st.session_state['theme'] = 'light' if is_light else 'dark'
        st.rerun()

    _is_dark_sb = st.session_state['theme'] == 'dark'
    _acc  = '#00f0ff'  if _is_dark_sb else '#1a365d'
    _txt  = '#c8d6e5'  if _is_dark_sb else '#2d3748'
    _dim  = '#8899aa'  if _is_dark_sb else '#4a5568'
    _ok   = '#64ffda'  if _is_dark_sb else '#276749'
    _warn = '#ffd166'  if _is_dark_sb else '#b7791f'
    _err  = '#ff6b6b'  if _is_dark_sb else '#c53030'
    _bar_bg = 'rgba(255,255,255,0.08)' if _is_dark_sb else 'rgba(0,0,0,0.08)'

    def _sb_label(text):
        st.markdown(
            f'<div style="color:{_acc}; font-size:0.68rem; letter-spacing:0.12em;'
            f' font-weight:700; text-transform:uppercase; margin:6px 0 4px 0;">'
            f'{text}</div>',
            unsafe_allow_html=True,
        )

    if not data_manager.is_data_loaded():
        st.divider()
        st.markdown(
            f'<div style="color:{_dim}; font-size:0.8rem; line-height:1.6;">'
            f'Load a <b style="color:{_txt}">.dat</b> or '
            f'<b style="color:{_txt}">.npz</b> file from the main page to begin.</div>',
            unsafe_allow_html=True,
        )
    else:
        pi     = data_manager.get_parser_info()
        result = data_manager.get_result()
        tc     = pi['type_counts']

        # ── File info ─────────────────────────────────────────────
        st.divider()
        _sb_label("File")
        st.markdown(
            f'<div style="color:{_txt}; font-size:0.82rem; font-weight:600;'
            f' word-break:break-all; line-height:1.4;">{pi["filename"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:{_dim}; font-size:0.75rem; margin-top:3px;">'
            f'{pi["file_size"]/1024**2:.1f} MB &nbsp;·&nbsp; '
            f'{pi["total_packets"]:,} packets &nbsp;·&nbsp; '
            f'{pi["parse_time_sec"]:.1f}s</div>',
            unsafe_allow_html=True,
        )

        # ── Time coverage ─────────────────────────────────────────
        _tr = st.session_state.get('time_range')
        if _tr:
            st.divider()
            _sb_label("Time Coverage")
            _t0 = datetime.fromtimestamp(_tr[0])
            _t1 = datetime.fromtimestamp(_tr[1])
            _dur = int(_tr[1] - _tr[0])
            _h, _m, _s = _dur // 3600, (_dur % 3600) // 60, _dur % 60
            _dur_str = (f'{_h}h {_m}m' if _h else f'{_m}m {_s}s') if _dur >= 60 else f'{_s}s'
            st.markdown(
                f'<div style="color:{_txt}; font-size:0.78rem; line-height:1.8;">'
                f'<span style="color:{_dim}">Start</span>&nbsp;'
                f'{_t0.strftime("%Y-%m-%d %H:%M")}<br>'
                f'<span style="color:{_dim}">End</span>&nbsp;&nbsp;&nbsp;'
                f'{_t1.strftime("%Y-%m-%d %H:%M")}<br>'
                f'<span style="color:{_dim}">Span</span>&nbsp;&nbsp;'
                f'{_dur_str}</div>',
                unsafe_allow_html=True,
            )

        # ── Packet breakdown ──────────────────────────────────────
        st.divider()
        _sb_label("Packets")
        _total = max(pi['total_packets'], 1)
        _ptype_colors = {
            'SPEC': '#00f0ff' if _is_dark_sb else '#1f77b4',
            'VNA':  '#a0ff00' if _is_dark_sb else '#2ca02c',
            'TEMP': '#ff9500',
        }
        _rows = []
        for _pt, _color in _ptype_colors.items():
            _cnt = tc.get(_pt, 0)
            _pct = _cnt / _total * 100
            _rows.append(
                f'<div style="margin:5px 0;">'
                f'  <div style="display:flex;justify-content:space-between;'
                f'       font-size:0.78rem;margin-bottom:2px;">'
                f'    <span style="color:{_dim}">{_pt}</span>'
                f'    <span style="color:{_txt};font-weight:600;">{_cnt:,}</span>'
                f'  </div>'
                f'  <div style="background:{_bar_bg};border-radius:2px;height:4px;">'
                f'    <div style="background:{_color};width:{_pct:.1f}%;'
                f'         height:4px;border-radius:2px;"></div>'
                f'  </div>'
                f'</div>'
            )
        _n_drop = len(pi['dropped_records'])
        if _n_drop > 0:
            _dpct = _n_drop / _total * 100
            _rows.append(
                f'<div style="margin:5px 0;">'
                f'  <div style="display:flex;justify-content:space-between;'
                f'       font-size:0.78rem;margin-bottom:2px;">'
                f'    <span style="color:{_dim}">DROPPED</span>'
                f'    <span style="color:{_err};font-weight:600;">{_n_drop:,}</span>'
                f'  </div>'
                f'  <div style="background:{_bar_bg};border-radius:2px;height:4px;">'
                f'    <div style="background:{_err};width:{_dpct:.1f}%;'
                f'         height:4px;border-radius:2px;"></div>'
                f'  </div>'
                f'</div>'
            )
        st.markdown(''.join(_rows), unsafe_allow_html=True)

        # ── Subsystem status ──────────────────────────────────────
        st.divider()
        _sb_label("Subsystems")
        _status_rows = []
        for _pt in ['SPEC', 'VNA', 'TEMP']:
            _key = _pt.lower()
            if _key in result and tc.get(_pt, 0) > 0:
                _dot_color, _status_text = _ok, f'{tc[_pt]:,} pkts'
            elif _key in result:
                _dot_color, _status_text = _warn, 'no valid data'
            else:
                _dot_color, _status_text = _err, 'no packets'
            _status_rows.append(
                f'<div style="display:flex;align-items:center;gap:6px;'
                f'     margin:5px 0;font-size:0.8rem;">'
                f'  <div style="width:7px;height:7px;border-radius:50%;'
                f'       background:{_dot_color};flex-shrink:0;'
                f'       box-shadow:0 0 5px {_dot_color}55;"></div>'
                f'  <span style="color:{_txt};font-weight:600;min-width:40px;">{_pt}</span>'
                f'  <span style="color:{_dim};">{_status_text}</span>'
                f'</div>'
            )
        st.markdown(''.join(_status_rows), unsafe_allow_html=True)

        # ── Actions ───────────────────────────────────────────────
        st.divider()
        if (st.session_state.get('last_parsed_path') is not None
                and st.button("Re-parse", use_container_width=True)):
            _start_async_parse(st.session_state['last_parsed_path'])
            st.rerun()


# ==============================================================
#  Main area — title
# ==============================================================

st.title("HONGMENG HIGH-FREQ SA")

# ==============================================================
#  Parse output panel
# ==============================================================

_pl = st.session_state.get('parse_log')
if _pl is not None:
    if not _pl['success']:
        with st.expander("PARSE OUTPUT", expanded=True):
            st.error(f"**LINK FAILED**\n\n```\n{_pl['error']}\n```")
            if _pl.get('traceback'):
                with st.expander("Full traceback"):
                    st.code(_pl['traceback'], language='python')
            _log_lines = _pl.get('parse_log_lines', [])
            if _log_lines:
                with st.expander(f"Parser log ({len(_log_lines)} lines)", expanded=True):
                    _render_parse_log(_log_lines)
    else:
        _pi = _pl['parser_info']
        _n_dropped  = len(_pi['dropped_records'])
        _n_errors   = _pi['error_count']
        _anomalies  = _pi.get('anomalies', {})
        _log_lines  = _pl.get('parse_log_lines', [])
        _n_anomaly  = sum(len(v) for v in _anomalies.values() if hasattr(v, '__len__'))
        _has_issues = _n_errors > 0 or _n_dropped > 0 or _n_anomaly > 0

        # Collapsed by default once data is loaded (summary metrics shown separately below)
        _expand_output = _has_issues or not data_manager.is_data_loaded()
        with st.expander("PARSE OUTPUT", expanded=_expand_output):
            st.caption(
                f"**{_pi['filename']}** · "
                f"{_pi['file_size'] / 1024 / 1024:.1f} MB · "
                f"{_pi['total_packets']} packets · "
                f"{_pi['parse_time_sec']:.2f}s"
            )
            # Parser log — inline
            if _log_lines:
                _n_warn = sum(1 for lv, _ in _log_lines if lv in ('WARNING', 'ERROR'))
                st.caption(f"Parser log · {len(_log_lines)} lines"
                           + (f" · {_n_warn} warning/error" if _n_warn else ""))
                _render_parse_log(_log_lines)

            # Errors / anomalies
            if _n_errors > 0:
                st.error(f"**{_n_errors} parse error(s)** encountered")
            if _n_dropped > 0:
                st.warning(f"**{_n_dropped} record(s) dropped**")
                with st.expander(f"Dropped record details ({_n_dropped})"):
                    for _rec in _pi['dropped_records'][:50]:
                        st.text(str(_rec))
                    if _n_dropped > 50:
                        st.caption(f"… and {_n_dropped - 50} more")
            for _cat, _items in _anomalies.items():
                _n = len(_items) if hasattr(_items, '__len__') else 0
                if _n > 0:
                    st.warning(f"**Anomaly [{ANOMALY_CATEGORIES.get(_cat, _cat)}]:** {_n} instance(s)")
            if not _has_issues:
                st.success("No errors or anomalies detected.")


# ==============================================================
#  Landing / data loaded
# ==============================================================

if not data_manager.is_data_loaded():
    _is_dark = st.session_state['theme'] == 'dark'
    st.markdown("### Load Data File")

    file_path_input = st.text_input(
        "File path",
        placeholder="/path/to/data.dat  or  /path/to/data.npz",
    )

    _valid_path = False
    _p_input = None
    if file_path_input:
        _p_input = Path(file_path_input.strip())
        if not _p_input.exists():
            st.error("File not found.")
        elif _p_input.suffix.lower() not in ('.dat', '.npz'):
            st.error("Unsupported file type. Use .dat or .npz")
        else:
            _valid_path = True
            _size_bytes = _p_input.stat().st_size
            st.caption(
                f"`{_p_input.name}` · {_size_bytes / 1024**2:.0f} MB · "
                + ("streaming" if _size_bytes >= 512 * 1024**2 else "in-memory")
            )

    _chunk_col, _ = st.columns([2, 3])
    with _chunk_col:
        _chunk_mb = st.select_slider(
            "Streaming chunk size", options=[32, 64, 128, 256, 512, 1024],
            value=st.session_state['parse_chunk_mb'],
            format_func=lambda x: f"{x} MB", key="chunk_slider",
        )
        st.session_state['parse_chunk_mb'] = _chunk_mb

    if _valid_path and _p_input is not None:
        if st.button("PARSE", type="primary"):
            if _p_input.suffix.lower() == '.npz':
                with st.spinner(f"Loading {_p_input.name} …"):
                    _start_sync_npz(str(_p_input))
                st.rerun()
            else:
                _start_async_parse(str(_p_input))
                st.rerun()

    if _pl is None:
        st.divider()
        _sub = '#8899aa' if _is_dark else '#6b7280'
        st.markdown(
            f'<div style="color:{_sub}; font-size:0.8rem; margin-bottom:12px; '
            f'letter-spacing:0.08em;">AVAILABLE SUBSYSTEMS</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**TEMPERATURE**\n\n25-point thermal monitoring with alerts")
        with c2:
            st.markdown("**SPECTRUM**\n\nSPEC visualization: 1D plots & waterfall")
        with c3:
            st.markdown("**S-PARAMETER**\n\nVNA S11: magnitude, phase, Smith chart")
        with c4:
            st.markdown("**LOAD .dat / .npz**\n\nPaste file path above and click PARSE")

else:
    # ── Summary metrics (single set — no duplicate with PARSE OUTPUT) ────
    parser_info = data_manager.get_parser_info()
    result      = data_manager.get_result()
    tc          = parser_info['type_counts']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SPEC", tc.get('SPEC', 0))
    with col2:
        st.metric("VNA", tc.get('VNA', 0))
    with col3:
        st.metric("TEMP", tc.get('TEMP', 0))
    with col4:
        st.metric("DROPPED", len(parser_info['dropped_records']),
                  delta=f"-{parser_info['error_count']} errors"
                        if parser_info['error_count'] else None,
                  delta_color="inverse")

    time_range = st.session_state.get('time_range')
    if time_range:
        t0 = datetime.fromtimestamp(time_range[0])
        t1 = datetime.fromtimestamp(time_range[1])
        st.caption(
            f"**{parser_info['filename']}** · "
            f"{parser_info['file_size'] / 1024 / 1024:.1f} MB · "
            f"{t0.strftime('%Y-%m-%d %H:%M:%S')} ~ {t1.strftime('%H:%M:%S')} · "
            f"parsed in {parser_info['parse_time_sec']:.1f}s"
        )

    # ── Packet count bar chart ────────────────────────────────────
    fig_bar = plot_utils.create_packet_count_bar(
        {k: v for k, v in tc.items() if k != 'UNKNOWN'},
        len(parser_info['dropped_records']),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Packet Status (inline) ────────────────────────────────────
    st.divider()

    # Device health row
    h1, h2, h3 = st.columns(3)
    for col, ptype in zip([h1, h2, h3], ['spec', 'vna', 'temp']):
        with col:
            if ptype in result:
                count = tc.get(ptype.upper(), 0)
                t_arr = result[ptype]['time']
                if len(t_arr) > 0 and count > 0:
                    dt = datetime.fromtimestamp(float(t_arr[-1]))
                    st.success(f"**{ptype.upper()}** · {count} pkts · last {dt.strftime('%H:%M:%S')}")
                else:
                    st.warning(f"**{ptype.upper()}**: no valid data")
            else:
                st.error(f"**{ptype.upper()}**: no packets")

    # Anomalies + sequence gaps + dropped (compact)
    anomalies = parser_info.get('anomalies', {})
    _anomaly_items = [(k, v) for k, v in anomalies.items()
                      if hasattr(v, '__len__') and len(v) > 0]

    all_gaps = []
    for ptype in ('spec', 'vna', 'temp'):
        if ptype in result:
            sub = result[ptype]
            all_gaps.extend(
                dp.compute_seq_gap_details(sub['seq'], sub['time'], ptype.upper()))

    dropped = parser_info['dropped_records']

    _has_any = _anomaly_items or all_gaps or dropped
    if not _has_any:
        st.success("No anomalies, sequence gaps, or dropped packets.")
    else:
        if _anomaly_items:
            for cat, messages in _anomaly_items:
                cat_name = ANOMALY_CATEGORIES.get(cat, cat)
                with st.expander(f"Anomaly: {cat_name} ({len(messages)})"):
                    for msg in messages:
                        st.markdown(f"- {msg}")
        if all_gaps:
            with st.expander(f"Sequence Gaps ({len(all_gaps)})"):
                st.dataframe(pd.DataFrame(all_gaps),
                             use_container_width=True, hide_index=True)
        if dropped:
            with st.expander(f"Dropped Packets ({len(dropped)})"):
                df_d = pd.DataFrame(dropped)
                if 'raw_hex' in df_d.columns:
                    df_d['raw_hex'] = df_d['raw_hex'].str[:60] + '…'
                st.dataframe(df_d, use_container_width=True, hide_index=True)

    # ── Load different file ───────────────────────────────────────
    st.divider()
    with st.expander("Load a different file"):
        _new_path = st.text_input("File path (.dat / .npz)",
                                  key="new_file_path_loaded",
                                  placeholder="/path/to/other_file.dat")
        if _new_path:
            _np2 = Path(_new_path.strip())
            if not _np2.exists():
                st.error("File not found.")
            elif _np2.suffix.lower() not in ('.dat', '.npz'):
                st.error("Unsupported file type.")
            else:
                st.caption(f"{_np2.stat().st_size / 1024**2:.0f} MB")
                _nc_col, _ = st.columns([2, 3])
                with _nc_col:
                    _new_chunk_mb = st.select_slider(
                        "Streaming chunk size", options=[32, 64, 128, 256, 512, 1024],
                        value=st.session_state['parse_chunk_mb'],
                        format_func=lambda x: f"{x} MB", key="chunk_slider",
                    )
                    st.session_state['parse_chunk_mb'] = _new_chunk_mb
                if st.button("PARSE NEW FILE", type="primary"):
                    if _np2.suffix.lower() == '.npz':
                        with st.spinner(f"Loading {_np2.name} …"):
                            _start_sync_npz(str(_np2))
                        st.rerun()
                    else:
                        _start_async_parse(str(_np2))
                        st.rerun()
