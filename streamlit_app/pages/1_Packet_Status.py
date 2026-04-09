"""
Page 1: Packet Status Dashboard
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

import data_manager
import data_processor as dp
import plot_utils
from config import ANOMALY_CATEGORIES
from theme import apply_theme

apply_theme()

st.title("Packet Status Overview")

if not data_manager.is_data_loaded():
    st.warning("No data loaded. Please upload a file from the main page.")
    st.stop()

parser_info = data_manager.get_parser_info()
result = data_manager.get_result()
tc = parser_info['type_counts']

# ==============================================================
#  Metrics row
# ==============================================================

cols = st.columns(5)
with cols[0]:
    st.metric("SPEC", tc.get('SPEC', 0))
with cols[1]:
    st.metric("VNA", tc.get('VNA', 0))
with cols[2]:
    st.metric("TEMP", tc.get('TEMP', 0))
with cols[3]:
    st.metric("UNKNOWN", tc.get('UNKNOWN', 0))
with cols[4]:
    st.metric("Dropped", len(parser_info['dropped_records']),
              delta=f"{parser_info['error_count']} errors" if parser_info['error_count'] else "0 errors",
              delta_color="inverse")

# ==============================================================
#  Packet count chart
# ==============================================================

fig_bar = plot_utils.create_packet_count_bar(
    {k: v for k, v in tc.items() if k != 'UNKNOWN'},
    len(parser_info['dropped_records'])
)
st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================
#  Device health status
# ==============================================================

st.subheader("Device Health")

health_cols = st.columns(3)
for i, ptype in enumerate(['spec', 'vna', 'temp']):
    with health_cols[i]:
        if ptype in result:
            count = tc.get(ptype.upper(), 0)
            t = result[ptype]['time']
            if len(t) > 0 and count > 0:
                latest = float(t[-1])
                from datetime import datetime
                dt = datetime.fromtimestamp(latest)
                st.success(f"**{ptype.upper()}**: OK ({count} pkts, last: {dt.strftime('%H:%M:%S')})")
            else:
                st.warning(f"**{ptype.upper()}**: No valid data")
        else:
            st.error(f"**{ptype.upper()}**: No packets received")

# ==============================================================
#  Anomaly detection results
# ==============================================================

st.subheader("Data Anomalies")
anomalies = parser_info.get('anomalies', {})

if not anomalies:
    st.success("No anomalies detected.")
else:
    for cat, messages in anomalies.items():
        cat_name = ANOMALY_CATEGORIES.get(cat, cat)
        with st.expander(f"{cat_name} ({len(messages)} items)", expanded=True):
            for msg in messages:
                st.markdown(f"- {msg}")

# ==============================================================
#  Sequence gap details
# ==============================================================

st.subheader("Sequence Gap Analysis")

all_gaps = []
for ptype in ('spec', 'vna', 'temp'):
    if ptype in result:
        sub = result[ptype]
        gaps = dp.compute_seq_gap_details(sub['seq'], sub['time'], ptype.upper())
        all_gaps.extend(gaps)

if all_gaps:
    df_gaps = pd.DataFrame(all_gaps)
    st.dataframe(df_gaps, use_container_width=True, hide_index=True)
else:
    st.success("No sequence gaps detected.")

# ==============================================================
#  Dropped packet details
# ==============================================================

st.subheader("Dropped Packets")

dropped = parser_info['dropped_records']
if dropped:
    with st.expander(f"Show {len(dropped)} dropped records"):
        df_dropped = pd.DataFrame(dropped)
        # Truncate raw_hex for display
        if 'raw_hex' in df_dropped.columns:
            df_dropped['raw_hex'] = df_dropped['raw_hex'].str[:60] + '...'
        st.dataframe(df_dropped, use_container_width=True, hide_index=True)
else:
    st.success("No dropped packets.")
