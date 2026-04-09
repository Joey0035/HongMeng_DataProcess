"""
plot_utils.py — Plotly chart factory functions with dark/light theme support.
All functions return plotly.graph_objects.Figure instances.
"""

import numpy as np
import plotly.graph_objects as go
from datetime import datetime


# ==============================================================
#  Theme system
# ==============================================================

_THEMES = {
    'dark': dict(
        colors=[
            '#00f0ff', '#ff3c7d', '#a0ff00', '#ff9500', '#bf5fff',
            '#00ff88', '#ff6b6b', '#40c4ff', '#ffd740', '#e040fb',
            '#64ffda', '#ff8a65', '#7c4dff', '#69f0ae', '#ff80ab',
            '#80d8ff', '#ffab40', '#b388ff', '#84ffff', '#ff5252',
        ],
        bg='#0a0e1a',
        paper='#0f1425',
        grid='rgba(0,240,255,0.08)',
        text='#c8d6e5',
        title_color='#00f0ff',
        axis_line='#4a5568',
        legend_bg='rgba(15,20,37,0.8)',
        legend_border='rgba(0,240,255,0.2)',
        smith_grid='rgba(0,240,255,0.15)',
        smith_circle='rgba(0,240,255,0.4)',
        smith_axis='rgba(0,240,255,0.25)',
        heatmap_scale='Inferno',
        phase_scale=[[0,'#0000aa'],[0.25,'#004488'],[0.5,'#222222'],[0.75,'#884400'],[1,'#aa0000']],
        bar_colors=['#00f0ff', '#a0ff00', '#ff9500', '#ff3c7d'],
        bar_line='rgba(0,240,255,0.3)',
        th_high='#ff3c7d',
        th_low='#40c4ff',
    ),
    'light': dict(
        colors=[
            '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
            '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
            '#aec7e8', '#ff9896', '#98df8a', '#ffbb78', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#9edae5', '#dbdb8d', '#c7c7c7',
        ],
        bg='#ffffff',
        paper='#f8f9fb',
        grid='rgba(0,0,0,0.06)',
        text='#2d3748',
        title_color='#1a365d',
        axis_line='#cbd5e0',
        legend_bg='rgba(255,255,255,0.95)',
        legend_border='rgba(0,0,0,0.1)',
        smith_grid='rgba(0,0,0,0.12)',
        smith_circle='rgba(0,0,0,0.3)',
        smith_axis='rgba(0,0,0,0.2)',
        heatmap_scale='Viridis',
        phase_scale='RdBu',
        bar_colors=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'],
        bar_line='rgba(0,0,0,0.15)',
        th_high='#e53e3e',
        th_low='#3182ce',
    ),
}

_current_theme = 'dark'


def set_theme(theme: str):
    global _current_theme
    _current_theme = theme if theme in _THEMES else 'dark'


def _t():
    """Get current theme dict."""
    return _THEMES[_current_theme]


def _pick_color(idx: int) -> str:
    return _t()['colors'][idx % len(_t()['colors'])]


# ==============================================================
#  Font helpers
# ==============================================================

def _font():
    t = _t()
    return dict(family='Consolas, JetBrains Mono, monospace', size=14, color=t['text'])

def _title_font():
    t = _t()
    return dict(family='Consolas, JetBrains Mono, monospace', size=18, color=t['title_color'])

def _axis_font():
    return dict(size=14, color=_t()['text'])

def _tick_font():
    return dict(size=12, color=_t()['text'])

def _legend_font():
    return dict(size=12, color=_t()['text'])


# ==============================================================
#  Themed layout builder
# ==============================================================

def _themed_layout(**kwargs) -> dict:
    t = _t()
    base = dict(
        font=_font(),
        title_font=_title_font(),
        paper_bgcolor=t['paper'],
        plot_bgcolor=t['bg'],
        legend=dict(
            font=_legend_font(),
            bgcolor=t['legend_bg'],
            bordercolor=t['legend_border'],
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor=t['grid'], zerolinecolor=t['axis_line'],
            title_font=_axis_font(), tickfont=_tick_font(), linecolor=t['axis_line'],
        ),
        yaxis=dict(
            gridcolor=t['grid'], zerolinecolor=t['axis_line'],
            title_font=_axis_font(), tickfont=_tick_font(), linecolor=t['axis_line'],
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    base.update(kwargs)
    return base


# ==============================================================
#  SPEC plots
# ==============================================================

def create_spec_1d_plot(split_data: dict, channel_idx: int,
                        freq_mhz: np.ndarray, channel_name: str,
                        time_range: tuple = None,
                        split_time: dict = None) -> go.Figure:
    fig = go.Figure()
    for i, (name, data) in enumerate(split_data.items()):
        if time_range and split_time and name in split_time:
            t = split_time[name]
            mask = (t >= time_range[0]) & (t <= time_range[1])
            d = data[mask]
        else:
            d = data
        if len(d) == 0:
            continue
        mean_spec = np.mean(d[:, channel_idx, :].astype(float), axis=0)
        dB = 10 * np.log10(np.abs(mean_spec).clip(1e-30))
        fig.add_trace(go.Scatter(
            x=freq_mhz, y=dB, mode='lines', name=name,
            line=dict(color=_pick_color(i), width=1.5),
        ))
    fig.update_layout(**_themed_layout(
        title=f'Time-Integrated Spectrum // {channel_name}',
        xaxis_title='Frequency (MHz)',
        yaxis_title='Power (dB)',
        height=520,
    ))
    return fig


def create_spec_waterfall(data_2d: np.ndarray, freq_mhz: np.ndarray,
                          times: list, title: str) -> go.Figure:
    dB = 10 * np.log10(np.abs(data_2d).astype(float).clip(1e-30))
    fig = go.Figure(data=go.Heatmap(
        z=dB, x=freq_mhz, y=times,
        colorscale=_t()['heatmap_scale'],
        colorbar=dict(title=dict(text='Power (dB)', font=_tick_font()), tickfont=_tick_font()),
    ))
    fig.update_layout(**_themed_layout(
        title=title, xaxis_title='Frequency (MHz)', yaxis_title='Time', height=620,
    ))
    return fig


# ==============================================================
#  VNA plots
# ==============================================================

def create_vna_magnitude_plot(split_data: dict, freq_mhz: np.ndarray,
                              time_range: tuple = None,
                              split_time: dict = None) -> go.Figure:
    fig = go.Figure()
    has_freq = freq_mhz is not None
    x = freq_mhz if has_freq else None
    for i, (name, data) in enumerate(split_data.items()):
        d = np.stack(data) if isinstance(data, list) and len(data) > 0 else data if not isinstance(data, list) else np.array([])
        if d.ndim < 2 or len(d) == 0:
            continue
        if time_range and split_time and name in split_time:
            t = split_time[name]
            d = d[(t >= time_range[0]) & (t <= time_range[1])]
        if len(d) == 0:
            continue
        if x is None:
            x = np.arange(d.shape[1])
        dB = 20 * np.log10(np.abs(d).mean(axis=0).clip(1e-30))
        fig.add_trace(go.Scatter(x=x, y=dB, mode='lines', name=name,
                                 line=dict(color=_pick_color(i), width=1.5)))
    fig.update_layout(**_themed_layout(
        title='|S11| // Time Integrated',
        xaxis_title='Frequency (MHz)' if has_freq else 'Freq Index',
        yaxis_title='|S11| (dB)', height=520,
    ))
    return fig


def create_vna_phase_plot(split_data: dict, freq_mhz: np.ndarray,
                          time_range: tuple = None,
                          split_time: dict = None) -> go.Figure:
    fig = go.Figure()
    has_freq = freq_mhz is not None
    x = freq_mhz if has_freq else None
    for i, (name, data) in enumerate(split_data.items()):
        d = np.stack(data) if isinstance(data, list) and len(data) > 0 else data if not isinstance(data, list) else np.array([])
        if d.ndim < 2 or len(d) == 0:
            continue
        if time_range and split_time and name in split_time:
            t = split_time[name]
            d = d[(t >= time_range[0]) & (t <= time_range[1])]
        if len(d) == 0:
            continue
        if x is None:
            x = np.arange(d.shape[1])
        phase = np.angle(d.mean(axis=0), deg=True)
        fig.add_trace(go.Scatter(x=x, y=phase, mode='lines', name=name,
                                 line=dict(color=_pick_color(i), width=1.5)))
    fig.update_layout(**_themed_layout(
        title='S11 Phase // Time Integrated',
        xaxis_title='Frequency (MHz)' if has_freq else 'Freq Index',
        yaxis_title='Phase (deg)', height=520,
    ))
    return fig


def create_vna_waterfall(data_2d: np.ndarray, freq_mhz: np.ndarray,
                         times: list, title: str, is_phase: bool = False) -> go.Figure:
    if isinstance(data_2d, list):
        data_2d = np.stack(data_2d)
    t = _t()
    if is_phase:
        z = np.angle(data_2d, deg=True)
        cbar_title, cscale = 'Phase (deg)', t['phase_scale']
    else:
        z = 20 * np.log10(np.abs(data_2d).astype(float).clip(1e-30))
        cbar_title, cscale = '|S11| (dB)', t['heatmap_scale']
    has_freq = freq_mhz is not None
    x = freq_mhz if has_freq else np.arange(z.shape[1])
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=times, colorscale=cscale,
        colorbar=dict(title=dict(text=cbar_title, font=_tick_font()), tickfont=_tick_font()),
    ))
    fig.update_layout(**_themed_layout(
        title=title,
        xaxis_title='Frequency (MHz)' if has_freq else 'Freq Index',
        yaxis_title='Time', height=620,
    ))
    return fig


# ==============================================================
#  Smith chart
# ==============================================================

def create_smith_chart(s11_dict: dict, freq_mhz: np.ndarray = None) -> go.Figure:
    fig = go.Figure()
    t = _t()
    theta = np.linspace(0, 2 * np.pi, 200)

    # Unit circle
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color=t['smith_circle'], width=1.5),
        showlegend=False, hoverinfo='skip'))

    # Constant resistance circles
    for r in [0, 0.2, 0.5, 1, 2, 5]:
        cx, cr = r / (r + 1), 1 / (r + 1)
        xc = cx + cr * np.cos(theta)
        yc = cr * np.sin(theta)
        mask = xc**2 + yc**2 <= 1.01
        xc[~mask], yc[~mask] = np.nan, np.nan
        fig.add_trace(go.Scatter(x=xc, y=yc, mode='lines',
                                 line=dict(color=t['smith_grid'], width=0.7),
                                 showlegend=False, hoverinfo='skip'))

    # Constant reactance arcs
    for xv in [0.2, 0.5, 1, 2, 5]:
        for s in [1, -1]:
            xa = 1.0 + (1/xv)*np.cos(theta)
            ya = s/xv + (1/xv)*np.sin(theta)
            mask = xa**2 + ya**2 <= 1.01
            xa[~mask], ya[~mask] = np.nan, np.nan
            fig.add_trace(go.Scatter(x=xa, y=ya, mode='lines',
                                     line=dict(color=t['smith_grid'], width=0.7),
                                     showlegend=False, hoverinfo='skip'))

    # Horizontal axis
    fig.add_trace(go.Scatter(x=[-1,1], y=[0,0], mode='lines',
                             line=dict(color=t['smith_axis'], width=0.7),
                             showlegend=False, hoverinfo='skip'))

    # Overlay S11 data
    for i, (name, s11) in enumerate(s11_dict.items()):
        if isinstance(s11, list):
            s11 = np.stack(s11)
        mean_s11 = s11.mean(axis=0) if s11.ndim == 2 else s11
        hover = []
        for k in range(len(mean_s11)):
            txt = f"{name}<br>|S11|={20*np.log10(max(abs(mean_s11[k]),1e-30)):.1f} dB"
            txt += f"<br>Phase={np.angle(mean_s11[k], deg=True):.1f} deg"
            if freq_mhz is not None and k < len(freq_mhz):
                txt += f"<br>Freq={freq_mhz[k]:.1f} MHz"
            hover.append(txt)
        c = _pick_color(i)
        fig.add_trace(go.Scatter(
            x=mean_s11.real, y=mean_s11.imag, mode='lines+markers', name=name,
            marker=dict(size=3, color=c), line=dict(color=c, width=1.5),
            text=hover, hoverinfo='text'))

    fig.update_layout(
        title=dict(text='Smith Chart', font=_title_font()),
        font=_font(), paper_bgcolor=t['paper'], plot_bgcolor=t['bg'],
        legend=dict(font=_legend_font(), bgcolor=t['legend_bg'],
                    bordercolor=t['legend_border'], borderwidth=1),
        xaxis=dict(range=[-1.15,1.15], scaleanchor='y', scaleratio=1,
                   showgrid=False, zeroline=False, title='Real',
                   title_font=_axis_font(), tickfont=_tick_font(), linecolor=t['axis_line']),
        yaxis=dict(range=[-1.15,1.15], showgrid=False, zeroline=False, title='Imag',
                   title_font=_axis_font(), tickfont=_tick_font(), linecolor=t['axis_line']),
        height=680, width=720, margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


# ==============================================================
#  Temperature plots
# ==============================================================

def create_temp_timeseries(temp_data: np.ndarray, time_arr: np.ndarray,
                           selected_points: list, thresholds: dict) -> go.Figure:
    fig = go.Figure()
    t = _t()
    dt_list = [datetime.fromtimestamp(float(v)) for v in time_arr]
    for i, (chip, ch, label) in enumerate(selected_points):
        fig.add_trace(go.Scatter(
            x=dt_list, y=temp_data[:, chip, ch], mode='lines', name=label,
            line=dict(color=_pick_color(i), width=1.5)))
    if thresholds.get('high') is not None:
        fig.add_hline(y=thresholds['high'], line_dash='dash', line_color=t['th_high'],
                      annotation_text=f"HIGH {thresholds['high']}°C",
                      annotation_font=dict(color=t['th_high'], size=13))
    if thresholds.get('low') is not None:
        fig.add_hline(y=thresholds['low'], line_dash='dash', line_color=t['th_low'],
                      annotation_text=f"LOW {thresholds['low']}°C",
                      annotation_font=dict(color=t['th_low'], size=13))
    fig.update_layout(**_themed_layout(
        title='Temperature Monitor', xaxis_title='Time',
        yaxis_title='Temperature (°C)', height=520))
    return fig


# ==============================================================
#  Packet status plots
# ==============================================================

def create_packet_count_bar(type_counts: dict, dropped: int) -> go.Figure:
    t = _t()
    types = list(type_counts.keys()) + ['Dropped']
    counts = list(type_counts.values()) + [dropped]
    fig = go.Figure(data=go.Bar(
        x=types, y=counts,
        marker_color=t['bar_colors'][:len(types)],
        marker_line=dict(color=t['bar_line'], width=1),
        text=counts, textposition='auto',
        textfont=dict(size=14, color=t['text']),
    ))
    fig.update_layout(**_themed_layout(
        title='Packet Count Overview', xaxis_title='Packet Type',
        yaxis_title='Count', height=380))
    return fig
