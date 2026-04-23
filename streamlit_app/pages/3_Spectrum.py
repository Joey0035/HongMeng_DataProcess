"""
Page 3: Spectrum Analysis (SPEC)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
from datetime import datetime, timedelta

import data_manager
import data_processor as dp
import plot_utils
from config import DTYPE_INDEX_MAP, DTYPE_LABELS, SOURCE_LABEL_MAP
from theme import apply_theme, sidebar_colors, sidebar_label, sidebar_kv, sidebar_file_info

apply_theme()

@st.cache_data
def _prepare_wf_source(d: np.ndarray, t_arr: np.ndarray,
                       t_start: float, t_end: float,
                       channel_idx: int, max_rows: int = 200):
    """时间过滤 + 通道提取 + 降采样，结果缓存，time_range/channel 变化时自动失效"""
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    d_f = d[mask]
    t_f = t_arr[mask]
    if len(d_f) == 0:
        return None, None
    wf_d = d_f[:, channel_idx, :]
    tl = dp.timestamps_to_datetime_strings(t_f)
    wf_d, tl, _ = dp.downsample_waterfall(wf_d, tl, max_rows=max_rows)
    return wf_d, tl

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

# Use pre-computed split (cached in session_state by store_in_session)
_cached = st.session_state.get('cache_spec_split')
if _cached is not None:
    split_data, split_time = _cached
else:
    split_data, split_time = dp.split_spec_by_src_with_time(result)
src_names = list(split_data.keys())

# ==============================================================
#  Controls
# ==============================================================

col_ctrl, col_cmap, col_time = st.columns([1, 1, 3])

with col_ctrl:
    channel_name = st.selectbox("Channel", DTYPE_LABELS)
    channel_idx = DTYPE_INDEX_MAP[channel_name]

with col_cmap:
    st.selectbox(
        "Colormap",
        plot_utils.COLORMAP_OPTIONS_MAG,
        index=plot_utils.COLORMAP_OPTIONS_MAG.index(
            st.session_state.get('global_colormap_mag', 'Inferno')),
        key='global_colormap_mag',
    )

with col_time:
    fft_src, fft_time = dp.get_fft_src_time(spec)
    t_min_dt = datetime.fromtimestamp(float(fft_time.min()))
    t_max_dt = datetime.fromtimestamp(float(fft_time.max()))
    time_range = st.slider(
        "Time Range",
        min_value=t_min_dt,
        max_value=t_max_dt,
        value=(t_min_dt, t_max_dt),
        format="MM/DD HH:mm",
        step=timedelta(minutes=30),
        key="spec_time_range",
    )

t_start = time_range[0].timestamp()
t_end = time_range[1].timestamp()

# --- Source selection: checkbox grid ---
st.divider()
st.caption("SOURCE SELECTION")
btn_col1, btn_col2, _ = st.columns([1, 1, 6])
with btn_col1:
    select_all = st.button("Select All", key="spec_sel_all", use_container_width=True)
with btn_col2:
    deselect_all = st.button("Deselect All", key="spec_desel_all", use_container_width=True)

if select_all:
    st.session_state['spec_src_state'] = {name: True for name in src_names}
    for name in src_names:
        st.session_state[f"spec_cb_{name}"] = True
    st.rerun()
if deselect_all:
    st.session_state['spec_src_state'] = {name: False for name in src_names}
    for name in src_names:
        st.session_state[f"spec_cb_{name}"] = False
    st.rerun()

# Initialize state dict if absent
if 'spec_src_state' not in st.session_state:
    st.session_state['spec_src_state'] = {name: True for name in src_names}
for name in src_names:
    if name not in st.session_state['spec_src_state']:
        st.session_state['spec_src_state'][name] = True
    # Pre-seed the widget key so checkbox doesn't see a value conflict
    _cb_key = f"spec_cb_{name}"
    if _cb_key not in st.session_state:
        st.session_state[_cb_key] = st.session_state['spec_src_state'][name]

n_cols = min(6, len(src_names))
grid_cols = st.columns(n_cols)
for idx, name in enumerate(src_names):
    with grid_cols[idx % n_cols]:
        checked = st.checkbox(name, key=f"spec_cb_{name}")
        st.session_state['spec_src_state'][name] = checked

selected_sources = [name for name in src_names if st.session_state['spec_src_state'].get(name, False)]

# Filter split data by selected sources
filtered_split = {k: v for k, v in split_data.items() if k in selected_sources}
filtered_time = {k: v for k, v in split_time.items() if k in selected_sources}

# ==============================================================
#  Sidebar
# ==============================================================

with st.sidebar:
    _is_light_sb = st.toggle("Day Mode",
                             value=(st.session_state['theme'] == 'light'),
                             key="theme_toggle")
    if (('light' if _is_light_sb else 'dark') != st.session_state['theme']):
        st.session_state['theme'] = 'light' if _is_light_sb else 'dark'
        st.rerun()

    c = sidebar_colors()
    sidebar_file_info(c)

    st.divider()
    sidebar_label("CURRENT VIEW", c)
    sidebar_kv("Channel", channel_name, c)
    sidebar_kv("Sources", f"{len(selected_sources)} / {len(src_names)}", c)
    sidebar_kv("Start",   time_range[0].strftime("%Y-%m-%d %H:%M"), c)
    sidebar_kv("End",     time_range[1].strftime("%Y-%m-%d %H:%M"), c)

# ==============================================================
#  Tabs
# ==============================================================

st.divider()

tab1, tab2, tab3 = st.tabs(["1D Spectrum", "Waterfall", "Waterfall Array"])

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
        wf_cmap = st.session_state.get('global_colormap_mag', 'Inferno')
        if wf_source in split_data:
            data_src = split_data[wf_source]
            time_src = split_time[wf_source]

            # Filter by time range
            mask = (time_src >= t_start) & (time_src <= t_end)
            data_filtered = data_src[mask]
            time_filtered = time_src[mask]

            if len(data_filtered) > 0:
                wf_data = data_filtered[:, channel_idx, :]
                _wf_fmin = float(freq_mhz[0])
                _wf_fmax = float(freq_mhz[-1])
                _wf_def_lo = max(_wf_fmin, 30.0)
                _wf_def_hi = min(_wf_fmax, 200.0)
                if _wf_def_lo >= _wf_def_hi:
                    _wf_def_lo, _wf_def_hi = _wf_fmin, _wf_fmax
                wf_freq_range = st.slider(
                    "Frequency Range (MHz)", min_value=_wf_fmin, max_value=_wf_fmax,
                    value=(_wf_def_lo, _wf_def_hi), key="spec_wf_freq",
                )
                time_labels = dp.timestamps_to_datetime_strings(time_filtered)
                wf_data, time_labels, ds = dp.downsample_waterfall(wf_data, time_labels)
                if ds:
                    st.info(f"Downsampled from {len(time_filtered)} to {len(time_labels)} time steps for display.")
                # Apply freq slice
                _f_mask = (freq_mhz >= wf_freq_range[0]) & (freq_mhz <= wf_freq_range[1])
                fig_wf = plot_utils.create_spec_waterfall(
                    wf_data[:, _f_mask], freq_mhz[_f_mask], time_labels,
                    f"Waterfall -- {wf_source} -- {channel_name}",
                    colorscale=wf_cmap,
                )
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.info("No data in the selected time range.")
    else:
        st.info("No sources available.")

with tab3:
    if filtered_split:
        freq_min_val, freq_max_val = float(freq_mhz[0]), float(freq_mhz[-1])
        _arr_def_lo = max(freq_min_val, 30.0)
        _arr_def_hi = min(freq_max_val, 200.0)
        if _arr_def_lo >= _arr_def_hi:
            _arr_def_lo, _arr_def_hi = freq_min_val, freq_max_val
        ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
        with ctrl1:
            spec_fr = st.slider(
                "Frequency Range (MHz)", min_value=freq_min_val, max_value=freq_max_val,
                value=(_arr_def_lo, _arr_def_hi), key="spec_wf_array_freq",
            )
        with ctrl2:
            n_grid_cols = st.select_slider("Columns", options=[2, 3, 4, 5, 6], value=3,
                                           key="spec_wf_array_cols")
        with ctrl3:
            per_page = st.select_slider("Sources/Page", options=[4, 6, 8, 12, 16], value=6,
                                        key="spec_wf_array_per_page")
        arr_cmap = st.session_state.get('global_colormap_mag', 'Inferno')

        # Prepare data: time-filter + channel select + downsample per source (cached)
        wf_array_data = {}
        wf_array_times = {}
        for name in selected_sources:
            if name not in split_data:
                continue
            wf_d, tl = _prepare_wf_source(
                split_data[name], split_time[name],
                t_start, t_end, channel_idx,
            )
            if wf_d is None:
                continue
            wf_array_data[name] = wf_d
            wf_array_times[name] = tl

        if wf_array_data:
            all_names = list(wf_array_data.keys())
            n_pages = max(1, (len(all_names) + per_page - 1) // per_page)
            page = 0
            if n_pages > 1:
                page = st.number_input(
                    f"Page (1-{n_pages})", min_value=1, max_value=n_pages,
                    value=1, key="spec_wf_array_page") - 1
                st.caption(f"Showing sources {page * per_page + 1}–{min((page + 1) * per_page, len(all_names))} of {len(all_names)}")
            page_names = all_names[page * per_page : (page + 1) * per_page]
            page_data = {k: wf_array_data[k] for k in page_names}
            page_times = {k: wf_array_times[k] for k in page_names}

            fig_arr = plot_utils.create_spec_waterfall_array(
                page_data, freq_mhz, page_times, channel_name,
                freq_range=spec_fr, n_cols=n_grid_cols, colorscale=arr_cmap,
            )
            st.plotly_chart(fig_arr, use_container_width=True)
        else:
            st.info("No data in the selected time range.")
    else:
        st.info("Select at least one source.")
