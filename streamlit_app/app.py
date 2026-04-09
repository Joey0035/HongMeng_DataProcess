"""
app.py — Main entry point for the HongMeng monitoring platform.
"""

import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="HongMeng Monitor",
    layout="wide",
    page_icon="\U0001f52d",
)

import data_manager
from config import DEFAULT_VNA_FREQ, DEFAULT_TEMP_THRESHOLDS
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

    load_mode = st.radio("Mode", ["Upload File", "Local Path"], horizontal=True)

    if load_mode == "Upload File":
        uploaded_file = st.file_uploader("Select .dat file", type=['dat'])
        if uploaded_file is not None:
            if st.button("PARSE", type="primary", use_container_width=True):
                progress_ph = st.empty()
                with st.spinner("Decoding telemetry..."):
                    result, parser_info = data_manager.parse_uploaded_file(
                        uploaded_file, progress_placeholder=progress_ph)
                    data_manager.store_in_session(result, parser_info)
                progress_ph.empty()
                st.success(
                    f"LINKED // {parser_info['total_packets']} packets "
                    f"in {parser_info['parse_time_sec']:.1f}s"
                )
    else:
        file_path = st.text_input("File path (.dat)")
        if file_path and st.button("PARSE", type="primary", use_container_width=True):
            progress_ph = st.empty()
            with st.spinner("Decoding telemetry..."):
                try:
                    result, parser_info = data_manager.parse_file_path(
                        file_path, progress_placeholder=progress_ph)
                    data_manager.store_in_session(result, parser_info)
                    progress_ph.empty()
                    st.success(
                        f"LINKED // {parser_info['total_packets']} packets "
                        f"in {parser_info['parse_time_sec']:.1f}s"
                    )
                except Exception as e:
                    progress_ph.empty()
                    st.error(f"LINK FAILED: {e}")

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

st.title("HONGMENG RADIO TELESCOPE")

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
