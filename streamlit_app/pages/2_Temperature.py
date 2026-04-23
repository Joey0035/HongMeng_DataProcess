"""
Page 2: Temperature Monitoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import data_manager
import data_processor as dp
import plot_utils
from config import TEMP_SENSOR_LABELS, TEMP_N_CHIPS, TEMP_CH_PER_CHIP
from theme import apply_theme, sidebar_colors, sidebar_label, sidebar_kv, sidebar_file_info

apply_theme()

@st.cache_data
def _cached_temp_stats(temp_data: np.ndarray, time_arr: np.ndarray,
                       selected_indices: tuple, t_start: float, t_end: float) -> dict:
    """cached wrapper — Streamlit 会对 numpy array 内容哈希，widget 改变时自动失效"""
    return dp.compute_temp_statistics(temp_data, time_arr, list(selected_indices), t_start, t_end)

st.title("Temperature Monitoring")

if not data_manager.is_data_loaded():
    st.warning("No data loaded. Please upload a file from the main page.")
    st.stop()

result = data_manager.get_result()
if 'temp' not in result:
    st.warning("No temperature data in this file.")
    st.stop()

temp = result['temp']
temp_data = temp['data']   # (n_pkt, 5, 5)
time_arr = temp['time']    # (n_pkt,)

thresholds = st.session_state.get('temp_thresholds', {'high': 100.0, 'low': -10.0})

# ==============================================================
#  Controls
# ==============================================================

# Controls: thresholds + time range in one compact row
_th_c1, _th_c2, _tc_time = st.columns([1, 1, 4])
with _th_c1:
    th_high = st.number_input("High Limit (°C)", value=float(thresholds['high']),
                               key="temp_th_high")
with _th_c2:
    th_low = st.number_input("Low Limit (°C)", value=float(thresholds['low']),
                              key="temp_th_low")
thresholds = {'high': th_high, 'low': th_low}
st.session_state['temp_thresholds'] = thresholds

with _tc_time:
    t_min_dt = datetime.fromtimestamp(float(time_arr[0]))
    t_max_dt = datetime.fromtimestamp(float(time_arr[-1]))
    time_range = st.slider(
        "Time Range",
        min_value=t_min_dt,
        max_value=t_max_dt,
        value=(t_min_dt, t_max_dt),
        format="MM/DD HH:mm",
        step=timedelta(minutes=30),
    )

t_start = time_range[0].timestamp()
t_end = time_range[1].timestamp()

# --- Sensor selection: checkbox grid by chip ---
st.divider()
st.caption("SENSOR SELECTION")
btn_col1, btn_col2, _ = st.columns([1, 1, 6])
with btn_col1:
    select_all = st.button("Select All", key="temp_sel_all", use_container_width=True)
with btn_col2:
    deselect_all = st.button("Deselect All", key="temp_desel_all", use_container_width=True)

if select_all:
    st.session_state['temp_src_state'] = {lbl: True for lbl in TEMP_SENSOR_LABELS}
    for lbl in TEMP_SENSOR_LABELS:
        st.session_state[f"temp_cb_{lbl}"] = True
    st.rerun()
if deselect_all:
    st.session_state['temp_src_state'] = {lbl: False for lbl in TEMP_SENSOR_LABELS}
    for lbl in TEMP_SENSOR_LABELS:
        st.session_state[f"temp_cb_{lbl}"] = False
    st.rerun()

if 'temp_src_state' not in st.session_state:
    st.session_state['temp_src_state'] = {lbl: True for lbl in TEMP_SENSOR_LABELS}

# Pre-seed widget keys to avoid value-conflict warning
for lbl in TEMP_SENSOR_LABELS:
    _tk = f"temp_cb_{lbl}"
    if _tk not in st.session_state:
        st.session_state[_tk] = st.session_state['temp_src_state'].get(lbl, True)

# Display as 5 columns (one per channel), 5 rows (one per chip)
grid_cols = st.columns(TEMP_CH_PER_CHIP)
for i in range(TEMP_N_CHIPS):
    for j in range(TEMP_CH_PER_CHIP):
        lbl = f"Chip{i}-Ch{j}"
        with grid_cols[j]:
            checked = st.checkbox(lbl, key=f"temp_cb_{lbl}")
            st.session_state['temp_src_state'][lbl] = checked

selected_labels = [lbl for lbl in TEMP_SENSOR_LABELS if st.session_state['temp_src_state'].get(lbl, False)]

# Convert labels to indices
selected_indices = [dp.label_to_chip_ch(lbl) for lbl in selected_labels]

# ==============================================================
#  Statistics panel
# ==============================================================

stats = _cached_temp_stats(temp_data, time_arr, tuple(selected_indices), t_start, t_end)

st.divider()
st.caption("STATISTICS")
stat_cols = st.columns(4)

# Find which probe has the global max and min
_pp = stats['per_point']
_max_sensor = max(_pp, key=lambda k: _pp[k]['max']) if _pp else None
_min_sensor = min(_pp, key=lambda k: _pp[k]['min']) if _pp else None

with stat_cols[0]:
    st.metric("Global Max", f"{stats['global_max']:.1f} °C")
    if _max_sensor:
        st.caption(f"@ {_max_sensor}")
with stat_cols[1]:
    st.metric("Global Min", f"{stats['global_min']:.1f} °C")
    if _min_sensor:
        st.caption(f"@ {_min_sensor}")
with stat_cols[2]:
    if selected_indices:
        avg_fluct = np.nanmean([v['fluctuation'] for v in stats['per_point'].values()])
        st.metric("Avg Fluctuation", f"{avg_fluct:.2f} °C")
    else:
        st.metric("Avg Fluctuation", "N/A")
with stat_cols[3]:
    if selected_indices:
        avg_mean = np.nanmean([v['mean'] for v in stats['per_point'].values()])
        st.metric("Avg Temperature", f"{avg_mean:.1f} °C")
    else:
        st.metric("Avg Temperature", "N/A")

# ==============================================================
#  Anomaly alerts
# ==============================================================

alerts = []
for chip, ch in selected_indices:
    label = f"Chip{chip}-Ch{ch}"
    pdata = stats['per_point'].get(label, {})
    if not pdata:
        continue
    if pdata['max'] > thresholds['high']:
        alerts.append((label, 'high', pdata['max']))
    if pdata['min'] < thresholds['low']:
        alerts.append((label, 'low', pdata['min']))

if alerts:
    st.subheader("Alerts")
    for label, kind, val in alerts:
        if kind == 'high':
            st.error(f"**{label}**: Max temp {val:.1f}°C exceeds threshold {thresholds['high']}°C")
        else:
            st.warning(f"**{label}**: Min temp {val:.1f}°C below threshold {thresholds['low']}°C")

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
    sidebar_kv("Start",   time_range[0].strftime("%Y-%m-%d %H:%M"), c)
    sidebar_kv("End",     time_range[1].strftime("%Y-%m-%d %H:%M"), c)
    sidebar_kv("Sensors", f"{len(selected_indices)} / {len(TEMP_SENSOR_LABELS)}", c)

    if stats and stats.get('per_point'):
        st.divider()
        sidebar_label("QUICK STATS", c)
        sidebar_kv("Max", f"{stats['global_max']:.1f} °C", c)
        sidebar_kv("Min", f"{stats['global_min']:.1f} °C", c)
        if alerts:
            st.markdown(
                f'<div style="color:{c["err"]}; font-size:0.78rem; margin-top:4px;">'
                f'{len(alerts)} alert{"s" if len(alerts) > 1 else ""} active</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="color:{c["ok"]}; font-size:0.78rem; margin-top:4px;">'
                f'No alerts</div>',
                unsafe_allow_html=True,
            )

# ==============================================================
#  Temperature time series chart
# ==============================================================

if selected_labels:
    st.divider()
    # Filter by time range
    mask = (time_arr >= t_start) & (time_arr <= t_end)
    filtered_data = temp_data[mask]
    filtered_time = time_arr[mask]

    selected_points = [(chip, ch, f"Chip{chip}-Ch{ch}")
                       for chip, ch in selected_indices]

    fig = plot_utils.create_temp_timeseries(
        filtered_data, filtered_time, selected_points, thresholds
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================
#  Per-sensor statistics table
# ==============================================================

if selected_indices:
    st.subheader("Per-Sensor Statistics")
    rows = []
    for label, pdata in stats['per_point'].items():
        rows.append({
            'Sensor': label,
            'Max (°C)': f"{pdata['max']:.2f}",
            'Min (°C)': f"{pdata['min']:.2f}",
            'Fluctuation (°C)': f"{pdata['fluctuation']:.2f}",
            'Mean (°C)': f"{pdata['mean']:.2f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
