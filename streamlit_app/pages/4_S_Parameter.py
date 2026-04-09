"""
Page 4: S-Parameter Analysis (VNA)
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
from theme import apply_theme

apply_theme()

st.title("S-Parameter Analysis (VNA)")

if not data_manager.is_data_loaded():
    st.warning("No data loaded. Please upload a file from the main page.")
    st.stop()

result = data_manager.get_result()
if 'vna' not in result:
    st.warning("No VNA data in this file.")
    st.stop()

vna = result['vna']
vna_freq_config = st.session_state.get('vna_freq_config', {})

# ==============================================================
#  Freq point selector
# ==============================================================

unique_nf = sorted(np.unique(vna['n_freq_per_sweep']).tolist())
unique_nf = [int(x) for x in unique_nf]

col_nf, col_time = st.columns([1, 3])

with col_nf:
    selected_nf = st.selectbox("Freq Points", unique_nf,
                               format_func=lambda x: f"{x} pts")

# Get frequency axis
freq_mhz = dp.get_vna_freq_mhz(selected_nf, vna_freq_config)

# Split by source for this n_freq
split_data, split_time = dp.split_vna_by_src_with_time(result, n_freq=selected_nf)
src_names = list(split_data.keys())

with col_time:
    sweep_times = dp.get_vna_sweep_times(vna)
    if len(sweep_times) > 0:
        t_min_dt = datetime.fromtimestamp(float(sweep_times.min()))
        t_max_dt = datetime.fromtimestamp(float(sweep_times.max()))
        time_range = st.slider(
            "Time Range",
            min_value=t_min_dt,
            max_value=t_max_dt,
            value=(t_min_dt, t_max_dt),
            format="HH:mm:ss",
            key="vna_time_range",
        )
        t_start = time_range[0].timestamp()
        t_end = time_range[1].timestamp()
    else:
        t_start, t_end = 0, float('inf')

# --- Source selection: checkbox grid ---
st.markdown("**Source Selection**")
btn_col1, btn_col2, _ = st.columns([1, 1, 6])
with btn_col1:
    select_all = st.button("Select All", key="vna_sel_all", use_container_width=True)
with btn_col2:
    deselect_all = st.button("Deselect All", key="vna_desel_all", use_container_width=True)

state_key = f'vna_src_state_{selected_nf}'
if select_all:
    st.session_state[state_key] = {name: True for name in src_names}
if deselect_all:
    st.session_state[state_key] = {name: False for name in src_names}

# Initialize: default all selected
if state_key not in st.session_state:
    st.session_state[state_key] = {name: True for name in src_names}
for name in src_names:
    if name not in st.session_state[state_key]:
        st.session_state[state_key][name] = True

n_cols = min(6, len(src_names))
grid_cols = st.columns(n_cols)
for idx, name in enumerate(src_names):
    with grid_cols[idx % n_cols]:
        st.session_state[state_key][name] = st.checkbox(
            name, value=st.session_state[state_key].get(name, True),
            key=f"vna_cb_{selected_nf}_{name}",
        )

selected_sources = [name for name in src_names if st.session_state[state_key].get(name, False)]

filtered_split = {k: v for k, v in split_data.items() if k in selected_sources}
filtered_time = {k: v for k, v in split_time.items() if k in selected_sources}

# ==============================================================
#  Tabs
# ==============================================================

tab1, tab2, tab3, tab4 = st.tabs(["|S11| Magnitude", "S11 Phase", "Waterfall", "Smith Chart"])

with tab1:
    if filtered_split:
        fig = plot_utils.create_vna_magnitude_plot(
            filtered_split, freq_mhz,
            time_range=(t_start, t_end), split_time=filtered_time,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one source.")

with tab2:
    if filtered_split:
        fig = plot_utils.create_vna_phase_plot(
            filtered_split, freq_mhz,
            time_range=(t_start, t_end), split_time=filtered_time,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one source.")

with tab3:
    if src_names:
        wf_source = st.selectbox("Waterfall Source",
                                 selected_sources if selected_sources else src_names,
                                 key="vna_wf_source")
        wf_mode = st.radio("Display", ["Magnitude (dB)", "Phase (deg)"],
                           horizontal=True, key="vna_wf_mode")

        if wf_source in split_data:
            data_src = split_data[wf_source]
            time_src = split_time[wf_source]

            # Convert list to array if needed
            if isinstance(data_src, list):
                data_src = np.stack(data_src)

            mask = (time_src >= t_start) & (time_src <= t_end)
            data_filtered = data_src[mask]
            time_filtered = time_src[mask]

            if len(data_filtered) > 0:
                time_labels = dp.timestamps_to_datetime_strings(time_filtered)
                data_ds, time_labels_ds, ds = dp.downsample_waterfall(data_filtered, time_labels)
                if ds:
                    st.info(f"Downsampled from {len(time_filtered)} to {len(time_labels_ds)} time steps for display.")
                is_phase = (wf_mode == "Phase (deg)")
                fig_wf = plot_utils.create_vna_waterfall(
                    data_ds, freq_mhz, time_labels_ds,
                    f"VNA Waterfall -- {wf_source} -- {'Phase' if is_phase else '|S11|'}",
                    is_phase=is_phase,
                )
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.info("No data in the selected time range.")
    else:
        st.info("No sources available.")

with tab4:
    if filtered_split:
        # For Smith chart, apply time filter
        smith_data = {}
        for name, data in filtered_split.items():
            if isinstance(data, list):
                data = np.stack(data)
            t = filtered_time.get(name)
            if t is not None:
                mask = (t >= t_start) & (t <= t_end)
                data = data[mask]
            if len(data) > 0:
                smith_data[name] = data

        if smith_data:
            fig_smith = plot_utils.create_smith_chart(smith_data, freq_mhz)
            st.plotly_chart(fig_smith, use_container_width=True)
        else:
            st.info("No data in the selected time range.")
    else:
        st.info("Select at least one source.")
