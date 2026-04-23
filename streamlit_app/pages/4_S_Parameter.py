"""
Page 4: S-Parameter Analysis (VNA)
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
from datetime import datetime, timedelta

import data_manager
import data_processor as dp
import plot_utils
from config import (
    CAL_SWITCH_PLANE, CAL_LNA_PLANE, CAL_LNA_CABLE_DEFAULTS, DEFAULT_VNA_FREQ,
)
from theme import apply_theme, sidebar_colors, sidebar_label, sidebar_kv, sidebar_file_info

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

col_nf, col_freq, col_time = st.columns([1, 2, 3])

with col_nf:
    selected_nf = st.selectbox("Freq Points", unique_nf,
                               format_func=lambda x: f"{x} pts")

# ── VNA Freq Config (inline, replaces sidebar widget) ────────────
vna_cfg = st.session_state.get('vna_freq_config', dict(DEFAULT_VNA_FREQ))
default_start, default_stop = vna_cfg.get(selected_nf, (0, 100))
with col_freq:
    _fc1, _fc2 = st.columns(2)
    with _fc1:
        f0 = st.number_input("Start (MHz)", value=float(default_start),
                             key=f"vna_f0_{selected_nf}")
    with _fc2:
        f1 = st.number_input("Stop (MHz)", value=float(default_stop),
                             key=f"vna_f1_{selected_nf}")
vna_cfg[selected_nf] = (f0, f1)
st.session_state['vna_freq_config'] = vna_cfg

# Get frequency axis from the inline config
freq_mhz = dp.get_vna_freq_mhz(selected_nf, vna_cfg)

# Use pre-computed split (cached in session_state by store_in_session)
_vna_splits = st.session_state.get('cache_vna_splits', {})
if selected_nf in _vna_splits:
    split_data, split_time = _vna_splits[selected_nf]
else:
    split_data, split_time = dp.split_vna_by_src_with_time(result, n_freq=selected_nf)
src_names = list(split_data.keys())

time_range = None
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
            format="MM/DD HH:mm",
            step=timedelta(minutes=30),
            key="vna_time_range",
        )
        t_start = time_range[0].timestamp()
        t_end = time_range[1].timestamp()
    else:
        t_start, t_end = 0, float('inf')

# ==============================================================
#  OSL Calibration Controls
# ==============================================================

cal_enabled = False
missing_sw = []
missing_lna = []
snp_file_path = None
cable_len = CAL_LNA_CABLE_DEFAULTS['length_m']
cable_vf = CAL_LNA_CABLE_DEFAULTS['velocity_factor']
cable_loss = CAL_LNA_CABLE_DEFAULTS['loss_db_per_m_per_ghz']

st.divider()
st.caption("OSL CALIBRATION")
_tog_col, _sw_col, _lna_col = st.columns([1, 2, 2])
with _tog_col:
    cal_enabled = st.toggle("OSL Cal", key="vna_cal_enabled")

if cal_enabled:
    missing_sw  = [v for k, v in CAL_SWITCH_PLANE.items() if v not in split_data]
    missing_lna = [v for k, v in CAL_LNA_PLANE.items()    if v not in split_data]
    with _sw_col:
        if missing_sw:
            st.warning(f"Switch: missing {', '.join(missing_sw)}")
        else:
            st.success("Switch standards: OK")
    with _lna_col:
        if missing_lna:
            st.info("LNA cal disabled (standards missing)")
        else:
            st.success("LNA standards: OK")

    # Cable offset row
    _cmode_col, _cparam1, _cparam2, _cparam3 = st.columns([1, 1, 1, 1])
    with _cmode_col:
        cable_mode = st.radio("Cable", ["Analytical", "SNP File"],
                              key="cable_mode")
    if cable_mode == "Analytical":
        with _cparam1:
            cable_len = st.number_input("Length (m)",
                value=CAL_LNA_CABLE_DEFAULTS['length_m'],
                format="%.4f", key="cable_len")
        with _cparam2:
            cable_vf = st.number_input("Velocity Factor",
                value=CAL_LNA_CABLE_DEFAULTS['velocity_factor'],
                format="%.3f", key="cable_vf")
        with _cparam3:
            cable_loss = st.number_input("Loss (dB/m/GHz)",
                value=CAL_LNA_CABLE_DEFAULTS['loss_db_per_m_per_ghz'],
                format="%.3f", key="cable_loss")
    else:
        snp_upload = st.file_uploader("Upload .s1p/.s2p",
                                      type=['s1p', 's2p'],
                                      key="snp_upload")
        if snp_upload is not None:
            suffix = Path(snp_upload.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(snp_upload.read())
                snp_file_path = tmp.name

# Apply calibration if enabled and standards available
cal_diag = None
if cal_enabled and not missing_sw and freq_mhz is not None:
    with st.spinner("Applying OSL calibration..."):
        try:
            cal_data, cal_diag = dp.apply_vna_calibration(
                split_data=split_data,
                freq_mhz=freq_mhz,
                switch_cal_keys=CAL_SWITCH_PLANE,
                lna_cal_keys=CAL_LNA_PLANE if not missing_lna else None,
                cable_length_m=cable_len,
                cable_vf=cable_vf,
                cable_loss=cable_loss,
                snp_file_path=snp_file_path,
            )
        finally:
            if snp_file_path:
                Path(snp_file_path).unlink(missing_ok=True)
    split_data = cal_data
    split_time = {k: v for k, v in split_time.items() if k in split_data}
    src_names = list(split_data.keys())

    n_cal = len(cal_diag.get('calibrated_sources', []))
    n_uncal = len(cal_diag.get('uncalibrated_sources', []))
    st.caption(f"Calibration applied: {n_cal} sources calibrated, {n_uncal} uncalibrated")

# ==============================================================
#  Source selection: checkbox grid
# ==============================================================

st.divider()
st.caption("SOURCE SELECTION")
btn_col1, btn_col2, _ = st.columns([1, 1, 6])
with btn_col1:
    select_all = st.button("Select All", key="vna_sel_all", use_container_width=True)
with btn_col2:
    deselect_all = st.button("Deselect All", key="vna_desel_all", use_container_width=True)

# Use a different state key for cal vs raw so that enabling cal resets defaults
_state_suffix = 'cal' if cal_enabled else 'raw'
state_key = f'vna_src_state_{selected_nf}_{_state_suffix}'
_cb_prefix = f"vna_cb_{selected_nf}_{_state_suffix}"

# Cal standard source names — hidden by default when calibration is enabled
_cal_std_names = set(CAL_SWITCH_PLANE.values()) | set(CAL_LNA_PLANE.values())

if select_all:
    st.session_state[state_key] = {name: True for name in src_names}
    for name in src_names:
        st.session_state[f"{_cb_prefix}_{name}"] = True
    st.rerun()
if deselect_all:
    st.session_state[state_key] = {name: False for name in src_names}
    for name in src_names:
        st.session_state[f"{_cb_prefix}_{name}"] = False
    st.rerun()

# Initialize: default all selected; cal standards unchecked when cal is enabled
if state_key not in st.session_state:
    st.session_state[state_key] = {
        name: (False if cal_enabled and name in _cal_std_names else True)
        for name in src_names
    }
for name in src_names:
    if name not in st.session_state[state_key]:
        st.session_state[state_key][name] = (
            False if cal_enabled and name in _cal_std_names else True
        )
# Remove stale keys no longer in src_names
st.session_state[state_key] = {
    k: v for k, v in st.session_state[state_key].items() if k in src_names
}
# Pre-seed widget keys to avoid value-conflict warning
for name in src_names:
    _cb_k = f"{_cb_prefix}_{name}"
    if _cb_k not in st.session_state:
        st.session_state[_cb_k] = st.session_state[state_key].get(name, True)

n_cols = min(6, len(src_names)) if src_names else 1
grid_cols = st.columns(n_cols)
for idx, name in enumerate(src_names):
    with grid_cols[idx % n_cols]:
        checked = st.checkbox(name, key=f"{_cb_prefix}_{name}")
        st.session_state[state_key][name] = checked

selected_sources = [name for name in src_names if st.session_state[state_key].get(name, False)]

filtered_split = {k: v for k, v in split_data.items() if k in selected_sources}
filtered_time = {k: v for k, v in split_time.items() if k in selected_sources}

# ==============================================================
#  Tabs
# ==============================================================


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
    sidebar_kv("Freq pts", f"{selected_nf}", c)
    if freq_mhz is not None:
        sidebar_kv("Range", f"{f0:.0f} → {f1:.0f} MHz", c)
    sidebar_kv("Sources", f"{len(selected_sources)} / {len(src_names)}", c)
    if time_range is not None:
        sidebar_kv("Start", time_range[0].strftime("%Y-%m-%d %H:%M"), c)
        sidebar_kv("End",   time_range[1].strftime("%Y-%m-%d %H:%M"), c)
    _cal_txt = ("On — OK"           if (cal_enabled and not missing_sw) else
                f"On — {len(missing_sw)} std missing" if cal_enabled else "Off")
    _cal_col = (c["ok"] if (cal_enabled and not missing_sw) else
                c["warn"] if cal_enabled else c["dim"])
    sidebar_kv("Cal", _cal_txt, c, val_color=_cal_col)

# ── Display settings ─────────────────────────────────────────────
st.divider()
_cmap_c1, _cmap_c2, _ = st.columns([1, 1, 4])
with _cmap_c1:
    st.selectbox(
        "Mag Colormap",
        plot_utils.COLORMAP_OPTIONS_MAG,
        index=plot_utils.COLORMAP_OPTIONS_MAG.index(
            st.session_state.get('global_colormap_mag', 'Inferno')),
        key='global_colormap_mag',
    )
with _cmap_c2:
    st.selectbox(
        "Phase Colormap",
        plot_utils.COLORMAP_OPTIONS_PHASE,
        index=plot_utils.COLORMAP_OPTIONS_PHASE.index(
            st.session_state.get('global_colormap_phase', 'RdBu')),
        key='global_colormap_phase',
    )

tab_labels = ["|S11| Magnitude", "S11 Phase", "Waterfall", "Waterfall Array", "Smith Chart"]
if cal_diag is not None:
    tab_labels.append("Cal Diagnostics")

tabs = st.tabs(tab_labels)

with tabs[0]:
    if filtered_split:
        fig = plot_utils.create_vna_magnitude_plot(
            filtered_split, freq_mhz,
            time_range=(t_start, t_end), split_time=filtered_time,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one source.")

with tabs[1]:
    if filtered_split:
        fig = plot_utils.create_vna_phase_plot(
            filtered_split, freq_mhz,
            time_range=(t_start, t_end), split_time=filtered_time,
        )
        # Overlay calibration standard model (ideal) curves
        if cal_diag is not None:
            import plotly.graph_objects as go
            std_self = cal_diag.get('standards_self_cal', {})
            n_existing = len(fig.data)
            for idx, (std_name, info) in enumerate(std_self.items()):
                expected = info['expected']
                xv = freq_mhz if freq_mhz is not None else np.arange(len(expected))
                fig.add_trace(go.Scatter(
                    x=xv, y=np.angle(expected, deg=True),
                    mode='lines', name=f"{std_name} (model)",
                    line=dict(color=plot_utils._pick_color(n_existing + idx), width=1.5, dash='dash'),
                    legendgroup='cal_model', legendgrouptitle_text='Ideal Standards',
                ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one source.")

with tabs[2]:
    if src_names:
        wf_source = st.selectbox("Waterfall Source",
                                 selected_sources if selected_sources else src_names,
                                 key="vna_wf_source")
        wf_mode = st.radio("Display", ["Magnitude (dB)", "Phase (deg)"],
                           horizontal=True, key="vna_wf_mode")
        vna_wf_cmap = (st.session_state.get('global_colormap_phase', 'RdBu')
                       if wf_mode == "Phase (deg)"
                       else st.session_state.get('global_colormap_mag', 'Inferno'))

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
                    is_phase=is_phase, colorscale=vna_wf_cmap,
                )
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.info("No data in the selected time range.")
    else:
        st.info("No sources available.")

with tabs[3]:
    if filtered_split:
        has_freq = freq_mhz is not None
        ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
        with ctrl1:
            if has_freq:
                freq_min_val, freq_max_val = float(freq_mhz[0]), float(freq_mhz[-1])
                vna_fr = st.slider(
                    "Frequency Range (MHz)", min_value=freq_min_val, max_value=freq_max_val,
                    value=(freq_min_val, freq_max_val), key="vna_wf_array_freq",
                )
            else:
                vna_fr = None
        with ctrl2:
            n_grid_cols = st.select_slider("Columns", options=[2, 3, 4, 5, 6], value=3,
                                           key="vna_wf_array_cols")
        with ctrl3:
            per_page = st.select_slider("Sources/Page", options=[4, 6, 8, 12, 16], value=6,
                                        key="vna_wf_array_per_page")
        vna_wf_mode = st.radio("Display", ["Magnitude (dB)", "Phase (deg)"],
                               horizontal=True, key="vna_wf_array_mode")
        vna_arr_cmap = (st.session_state.get('global_colormap_phase', 'RdBu')
                        if vna_wf_mode == "Phase (deg)"
                        else st.session_state.get('global_colormap_mag', 'Inferno'))

        wf_arr_data = {}
        wf_arr_times = {}
        for name in selected_sources:
            if name not in split_data:
                continue
            d = split_data[name]
            if isinstance(d, list):
                d = np.stack(d)
            t_arr = split_time.get(name)
            if t_arr is not None:
                mask = (t_arr >= t_start) & (t_arr <= t_end)
                d = d[mask]
                t_arr = t_arr[mask]
            if len(d) == 0:
                continue
            tl = dp.timestamps_to_datetime_strings(t_arr) if t_arr is not None else list(range(len(d)))
            d, tl, _ = dp.downsample_waterfall(d, tl, max_rows=200)
            wf_arr_data[name] = d
            wf_arr_times[name] = tl

        if wf_arr_data:
            all_names = list(wf_arr_data.keys())
            n_pages = max(1, (len(all_names) + per_page - 1) // per_page)
            page = 0
            if n_pages > 1:
                page = st.number_input(
                    f"Page (1-{n_pages})", min_value=1, max_value=n_pages,
                    value=1, key="vna_wf_array_page") - 1
                st.caption(f"Showing sources {page * per_page + 1}–{min((page + 1) * per_page, len(all_names))} of {len(all_names)}")
            page_names = all_names[page * per_page : (page + 1) * per_page]
            page_data = {k: wf_arr_data[k] for k in page_names}
            page_times = {k: wf_arr_times[k] for k in page_names}

            is_phase = (vna_wf_mode == "Phase (deg)")
            fig_arr = plot_utils.create_vna_waterfall_array(
                page_data, freq_mhz, page_times,
                is_phase=is_phase, freq_range=vna_fr, n_cols=n_grid_cols,
                colorscale=vna_arr_cmap,
            )
            st.plotly_chart(fig_arr, use_container_width=True)
        else:
            st.info("No data in the selected time range.")
    else:
        st.info("Select at least one source.")

with tabs[4]:
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

# ==============================================================
#  Cal Diagnostics tab (only when calibration is active)
# ==============================================================

if cal_diag is not None:
    with tabs[5]:
        import plotly.graph_objects as go

        std_self = cal_diag.get('standards_self_cal', {})
        if not std_self:
            st.info("No calibration diagnostics available.")
        else:
            st.subheader("Standards Self-Calibration Check")
            st.markdown(
                "Each standard is calibrated using its own plane's error terms. "
                "Ideally, the calibrated result should match the known model value exactly."
            )

            x = freq_mhz if freq_mhz is not None else None
            x_label = "Frequency (MHz)" if freq_mhz is not None else "Freq Index"

            # 预计算每个标准的 cal_mean，避免在三个循环中各算一次
            cal_means = {
                std_name: (np.mean(info['calibrated'], axis=0)
                           if info['calibrated'].ndim == 2 else info['calibrated'])
                for std_name, info in std_self.items()
            }

            # |Γ| magnitude plot
            fig_mag = go.Figure()
            for idx, (std_name, info) in enumerate(std_self.items()):
                cal_mean = cal_means[std_name]
                expected = info['expected']
                xv = x if x is not None else np.arange(len(cal_mean))

                fig_mag.add_trace(go.Scatter(
                    x=xv, y=20 * np.log10(np.abs(cal_mean).clip(1e-30)),
                    mode='lines', name=f"{std_name} (cal)",
                    line=dict(color=plot_utils._pick_color(idx * 2)),
                ))
                fig_mag.add_trace(go.Scatter(
                    x=xv, y=20 * np.log10(np.abs(expected).clip(1e-30)),
                    mode='lines', name=f"{std_name} (model)",
                    line=dict(dash='dash', color=plot_utils._pick_color(idx * 2 + 1)),
                ))

            fig_mag.update_layout(**plot_utils._themed_layout(
                title="Standards Self-Cal: |Γ| (dB)",
                xaxis_title=x_label, yaxis_title="|Γ| (dB)", height=520,
            ))
            st.plotly_chart(fig_mag, use_container_width=True)

            # Residual error plot
            fig_err = go.Figure()
            for idx, (std_name, info) in enumerate(std_self.items()):
                cal_mean = cal_means[std_name]
                expected = info['expected']
                residual = np.abs(cal_mean - expected)
                xv = x if x is not None else np.arange(len(residual))

                fig_err.add_trace(go.Scatter(
                    x=xv, y=residual,
                    mode='lines', name=std_name,
                    line=dict(color=plot_utils._pick_color(idx)),
                ))

            fig_err.update_layout(**plot_utils._themed_layout(
                title="Standards Self-Cal: Residual |Γ_cal - Γ_model|",
                xaxis_title=x_label, yaxis_title="Residual (linear)", height=520,
                yaxis=dict(type='log'),
            ))
            st.plotly_chart(fig_err, use_container_width=True)

            # Per-standard stats table
            rows = []
            for std_name, info in std_self.items():
                cal_mean = cal_means[std_name]
                expected = info['expected']
                residual = np.abs(cal_mean - expected)
                rows.append({
                    'Standard': std_name,
                    'Mean Residual': f"{np.mean(residual):.3e}",
                    'Max Residual': f"{np.max(residual):.3e}",
                    'Expected |Γ|': f"{np.mean(np.abs(expected)):.4f}",
                })
            st.dataframe(rows, use_container_width=True)
