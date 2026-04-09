"""
Page 3: Spectrum Analysis (SPEC)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
from datetime import datetime

import data_manager
import data_processor as dp
import plot_utils
from config import DTYPE_INDEX_MAP, DTYPE_LABELS, SOURCE_LABEL_MAP
from theme import apply_theme

apply_theme()

st.title("Spectrum Analysis (SPEC)")

if not data_manager.is_data_loaded():
    st.warning("No data loaded. Please upload a file from the main page.")
    st.stop()

result = data_manager.get_result()
if 'spec' not in result:
    st.warning("No SPEC data in this file.")
    st.stop()

spec = result['spec']
freq_mhz = dp.get_spec_freq_mhz()

# Split by source (with time)
split_data, split_time = dp.split_spec_by_src_with_time(result)
src_names = list(split_data.keys())

# ==============================================================
#  Controls
# ==============================================================

col_ctrl, col_time = st.columns([1, 2])

with col_ctrl:
    channel_name = st.selectbox("Channel", DTYPE_LABELS)
    channel_idx = DTYPE_INDEX_MAP[channel_name]

with col_time:
    fft_src, fft_time = dp.get_fft_src_time(spec)
    t_min_dt = datetime.fromtimestamp(float(fft_time.min()))
    t_max_dt = datetime.fromtimestamp(float(fft_time.max()))
    time_range = st.slider(
        "Time Range",
        min_value=t_min_dt,
        max_value=t_max_dt,
        value=(t_min_dt, t_max_dt),
        format="HH:mm:ss",
        key="spec_time_range",
    )

t_start = time_range[0].timestamp()
t_end = time_range[1].timestamp()

# --- Source selection: checkbox grid ---
st.markdown("**Source Selection**")
btn_col1, btn_col2, _ = st.columns([1, 1, 6])
with btn_col1:
    select_all = st.button("Select All", key="spec_sel_all", use_container_width=True)
with btn_col2:
    deselect_all = st.button("Deselect All", key="spec_desel_all", use_container_width=True)

if select_all:
    st.session_state['spec_src_state'] = {name: True for name in src_names}
if deselect_all:
    st.session_state['spec_src_state'] = {name: False for name in src_names}

# Initialize: default all selected
if 'spec_src_state' not in st.session_state:
    st.session_state['spec_src_state'] = {name: True for name in src_names}
# Sync if source list changed
for name in src_names:
    if name not in st.session_state['spec_src_state']:
        st.session_state['spec_src_state'][name] = True

n_cols = min(6, len(src_names))
grid_cols = st.columns(n_cols)
for idx, name in enumerate(src_names):
    with grid_cols[idx % n_cols]:
        st.session_state['spec_src_state'][name] = st.checkbox(
            name, value=st.session_state['spec_src_state'].get(name, True),
            key=f"spec_cb_{name}",
        )

selected_sources = [name for name in src_names if st.session_state['spec_src_state'].get(name, False)]

# Filter split data by selected sources
filtered_split = {k: v for k, v in split_data.items() if k in selected_sources}
filtered_time = {k: v for k, v in split_time.items() if k in selected_sources}

# ==============================================================
#  Tabs
# ==============================================================

tab1, tab2 = st.tabs(["1D Spectrum", "Waterfall"])

with tab1:
    if filtered_split:
        fig = plot_utils.create_spec_1d_plot(
            filtered_split, channel_idx, freq_mhz, channel_name,
            time_range=(t_start, t_end), split_time=filtered_time,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one source.")

with tab2:
    if src_names:
        wf_source = st.selectbox("Waterfall Source", selected_sources if selected_sources else src_names,
                                 key="wf_source")
        if wf_source in split_data:
            data_src = split_data[wf_source]
            time_src = split_time[wf_source]

            # Filter by time range
            mask = (time_src >= t_start) & (time_src <= t_end)
            data_filtered = data_src[mask]
            time_filtered = time_src[mask]

            if len(data_filtered) > 0:
                wf_data = data_filtered[:, channel_idx, :]
                time_labels = dp.timestamps_to_datetime_strings(time_filtered)
                wf_data, time_labels, ds = dp.downsample_waterfall(wf_data, time_labels)
                if ds:
                    st.info(f"Downsampled from {len(time_filtered)} to {len(time_labels)} time steps for display.")
                fig_wf = plot_utils.create_spec_waterfall(
                    wf_data, freq_mhz, time_labels,
                    f"Waterfall -- {wf_source} -- {channel_name}"
                )
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.info("No data in the selected time range.")
    else:
        st.info("No sources available.")
