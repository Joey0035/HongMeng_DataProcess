"""
theme.py — Shared theme CSS and application helper.

Called by app.py and every page to ensure consistent theme across all views.
"""

import streamlit as st
import plot_utils

# ==============================================================
#  Global CSS (theme-independent)
# ==============================================================

_GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', 'Consolas', monospace; }
button[data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.9rem; letter-spacing: 0.04em; }
[data-testid="stAlert"] { border-radius: 6px; border-left-width: 4px; }
.stCheckbox label span { font-size: 0.85rem; }
[data-testid="stSlider"] { padding-top: 0.5rem; }
/* ---- Section label chip ---- */
.section-chip {
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 0.7rem; letter-spacing: 0.1em; font-weight: 700;
    text-transform: uppercase; margin-bottom: 8px;
}
"""

# ==============================================================
#  Dark theme CSS
# ==============================================================

_DARK_CSS = """
/* ---- Streamlit CSS variables ---- */
:root {
    --primary-color: #00f0ff !important;
    --background-color: #0a0e1a !important;
    --secondary-background-color: #111827 !important;
    --text-color: #c8d6e5 !important;
}
/* ---- Main background glow ---- */
.stApp {
    background: linear-gradient(170deg, #0a0e1a 0%, #0d1526 40%, #0a0e1a 100%) !important;
    color: #c8d6e5 !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
.stMain, .main .block-container {
    background-color: transparent !important;
    color: #c8d6e5 !important;
}
[data-testid="stHeader"] {
    background-color: rgba(10,14,26,0.8) !important;
}
/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%) !important;
    border-right: 1px solid rgba(0,240,255,0.1);
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #c8d6e5 !important;
}
/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(0,240,255,0.05) 0%, rgba(15,20,37,0.8) 100%) !important;
    border: 1px solid rgba(0,240,255,0.15);
    border-radius: 8px; padding: 12px 16px;
    box-shadow: 0 0 15px rgba(0,240,255,0.05);
}
[data-testid="stMetricLabel"] {
    color: #00f0ff !important; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem; font-weight: 700;
    color: #e8f0fe !important; text-shadow: 0 0 8px rgba(0,240,255,0.3);
}
[data-testid="stMetricDelta"] { color: #a0aec0 !important; }
/* ---- Buttons ---- */
.stButton > button {
    border: 1px solid rgba(0,240,255,0.3); color: #00f0ff !important;
    background: rgba(0,240,255,0.05) !important; transition: all 0.3s ease;
}
.stButton > button:hover {
    border-color: #00f0ff; background: rgba(0,240,255,0.15) !important;
    box-shadow: 0 0 12px rgba(0,240,255,0.2);
}
button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,240,255,0.2), rgba(0,240,255,0.1)) !important;
    border-color: #00f0ff !important; box-shadow: 0 0 20px rgba(0,240,255,0.15);
}
/* ---- Headings ---- */
h1 { color: #00f0ff !important; text-shadow: 0 0 20px rgba(0,240,255,0.3); letter-spacing: 0.05em; }
h2, h3 { color: #64ffda !important; letter-spacing: 0.03em; }
/* ---- Body text ---- */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p {
    color: #c8d6e5 !important;
}
/* ---- Inputs ---- */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background-color: #111827 !important; color: #c8d6e5 !important;
    border-color: #2d3748 !important;
}
[data-baseweb="select"] > div {
    background-color: #111827 !important; color: #c8d6e5 !important;
    border-color: #2d3748 !important;
}
[data-baseweb="select"] span { color: #c8d6e5 !important; }
[data-baseweb="popover"], [data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] li, [data-baseweb="menu"], [data-baseweb="menu"] li {
    background-color: #111827 !important; color: #c8d6e5 !important;
}
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover {
    background-color: #1a2332 !important;
}
/* ---- Widget labels ---- */
.stRadio label, .stCheckbox label, .stToggle label,
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
    color: #c8d6e5 !important;
}
/* ---- Tabs ---- */
button[data-baseweb="tab"] { color: #8899aa !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #00f0ff !important; }
/* ---- File uploader ---- */
[data-testid="stFileUploader"], [data-testid="stFileUploader"] section {
    background-color: #111827 !important; border-color: #2d3748 !important;
}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: #8899aa !important; }
/* ---- Slider ---- */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #8899aa !important; }
/* ---- Expander ---- */
details {
    border: 1px solid rgba(0,240,255,0.15) !important;
    border-radius: 6px; margin-bottom: 4px;
}
details summary {
    background: rgba(0,240,255,0.06) !important;
    padding: 8px 12px !important; border-radius: 5px;
    cursor: pointer;
}
details summary:hover { background: rgba(0,240,255,0.12) !important; }
details[open] summary { border-radius: 5px 5px 0 0; }
details[open] { background-color: rgba(10,16,32,0.6) !important; }
details summary span, details summary p,
[data-testid="stExpander"] summary span {
    color: #c8d6e5 !important; font-weight: 500;
}
/* ---- DataFrame ---- */
[data-testid="stDataFrame"] { border: 1px solid rgba(0,240,255,0.1); border-radius: 6px; }
/* ---- Misc ---- */
hr { border-color: rgba(0,240,255,0.1) !important; }
/* ---- Alert ---- */
[data-testid="stAlert"] p, [data-testid="stAlert"] span { color: inherit !important; }
/* ---- Parse log terminal ---- */
.parse-log-box {
    background: #070b14; border: 1px solid rgba(0,240,255,0.12);
    border-radius: 6px; padding: 10px 14px;
    max-height: 420px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; line-height: 1.65;
}
.parse-log-box .ll-INFO  { color: #64ffda; }
.parse-log-box .ll-WARN  { color: #ffd166; }
.parse-log-box .ll-ERROR { color: #ff6b6b; }
.parse-log-box .ll-DEBUG { color: #556677; }
/* ---- Status badge ---- */
.linked-badge {
    display:inline-block; padding:4px 10px; border-radius:4px;
    background:rgba(100,255,218,0.1); border:1px solid rgba(100,255,218,0.3);
    color:#64ffda !important; font-size:0.8rem; letter-spacing:0.06em;
}
.section-chip { background:rgba(0,240,255,0.08); color:#00f0ff !important; }
"""

# ==============================================================
#  Light theme CSS
# ==============================================================

_LIGHT_CSS = """
/* ---- Streamlit CSS variables (override dark config.toml) ---- */
:root {
    --primary-color: #1f77b4 !important;
    --background-color: #f7f8fc !important;
    --secondary-background-color: #e8ecf1 !important;
    --text-color: #2d3748 !important;
}
/* ---- Main background ---- */
.stApp {
    background: linear-gradient(170deg, #f7f8fc 0%, #eef1f8 40%, #f7f8fc 100%) !important;
    color: #2d3748 !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
.stMain, .main .block-container {
    background-color: transparent !important;
    color: #2d3748 !important;
}
[data-testid="stHeader"] {
    background-color: rgba(247,248,252,0.9) !important;
}
/* Force all text dark */
.stApp p, .stApp span, .stApp label, .stApp div:not([data-testid="stAlert"]),
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"], .stCaption {
    color: #2d3748 !important;
}
/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f2f6 0%, #e8ecf1 100%) !important;
    border-right: 1px solid rgba(0,0,0,0.08);
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #2d3748 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: transparent !important;
}
/* ---- Inputs, selects, number inputs ---- */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background-color: #ffffff !important; color: #2d3748 !important; border-color: #cbd5e0 !important;
}
[data-baseweb="select"] { background-color: #ffffff !important; }
[data-baseweb="select"] > div {
    background-color: #ffffff !important; color: #2d3748 !important; border-color: #cbd5e0 !important;
}
[data-baseweb="select"] span { color: #2d3748 !important; }
/* Dropdown popover */
[data-baseweb="popover"], [data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] li, [data-baseweb="menu"], [data-baseweb="menu"] li {
    background-color: #ffffff !important; color: #2d3748 !important;
}
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover {
    background-color: #edf2f7 !important;
}
/* ---- File uploader ---- */
[data-testid="stFileUploader"], [data-testid="stFileUploader"] section {
    background-color: #ffffff !important; color: #2d3748 !important; border-color: #cbd5e0 !important;
}
[data-testid="stFileUploader"] button { color: #2b6cb0 !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: #4a5568 !important; }
/* ---- Radio, checkbox, toggle labels ---- */
.stRadio label, .stCheckbox label, .stToggle label,
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
    color: #2d3748 !important;
}
/* ---- Slider ---- */
[data-testid="stSlider"] [data-testid="stWidgetLabel"] { color: #2d3748 !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #4a5568 !important; }
[data-baseweb="slider"] [data-testid="stThumbValue"] { color: #2d3748 !important; }
/* ---- Tabs ---- */
button[data-baseweb="tab"] { color: #4a5568 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1a365d !important; }
[data-baseweb="tab-highlight"] { background-color: #1f77b4 !important; }
/* ---- Expander ---- */
details {
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 6px; margin-bottom: 4px;
}
details summary {
    background: rgba(31,119,180,0.07) !important;
    padding: 8px 12px !important; border-radius: 5px;
    cursor: pointer;
}
details summary:hover { background: rgba(31,119,180,0.14) !important; }
details[open] summary { border-radius: 5px 5px 0 0; }
details[open] { background-color: rgba(255,255,255,0.85) !important; }
details summary span, details summary p,
[data-testid="stExpander"] summary span {
    color: #1a365d !important; font-weight: 500;
}
/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(31,119,180,0.05) 0%, rgba(255,255,255,0.95) 100%) !important;
    border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] {
    color: #1a365d !important; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1a202c !important; }
[data-testid="stMetricDelta"] { color: #4a5568 !important; }
/* ---- Buttons ---- */
.stButton > button {
    border: 1px solid rgba(31,119,180,0.4); color: #1a365d !important;
    background: rgba(31,119,180,0.05) !important; transition: all 0.3s ease;
}
.stButton > button:hover {
    border-color: #1f77b4; background: rgba(31,119,180,0.12) !important;
    box-shadow: 0 2px 8px rgba(31,119,180,0.15);
}
button[kind="primary"] {
    background: linear-gradient(135deg, rgba(31,119,180,0.15), rgba(31,119,180,0.08)) !important;
    border-color: #1f77b4 !important; box-shadow: 0 2px 10px rgba(31,119,180,0.12);
    color: #1a365d !important;
}
/* ---- Headings ---- */
h1 { color: #1a365d !important; letter-spacing: 0.05em; }
h2, h3 { color: #2b6cb0 !important; letter-spacing: 0.03em; }
/* ---- Alert ---- */
[data-testid="stAlert"] p, [data-testid="stAlert"] span { color: inherit !important; }
/* ---- DataFrame / tables ---- */
[data-testid="stDataFrame"] { border: 1px solid rgba(0,0,0,0.08); border-radius: 6px; }
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    --gdg-bg-cell: #ffffff !important; --gdg-bg-header: #f7f8fc !important;
    --gdg-text-dark: #2d3748 !important; --gdg-text-header: #1a365d !important;
    --gdg-border-color: #e2e8f0 !important;
}
/* ---- Toggle switch track color ---- */
[data-testid="stToggle"] > label > div[role="checkbox"] {
    background-color: #cbd5e0 !important;
}
[data-testid="stToggle"] > label > div[role="checkbox"][aria-checked="true"] {
    background-color: #1f77b4 !important;
}
/* ---- Misc ---- */
hr { border-color: rgba(0,0,0,0.08) !important; }
.stSpinner > div { color: #2b6cb0 !important; }
/* ---- Plotly chart containers ---- */
.stPlotlyChart { background-color: transparent !important; }
/* ---- Parse log terminal ---- */
.parse-log-box {
    background: #f0f2f8; border: 1px solid rgba(0,0,0,0.1);
    border-radius: 6px; padding: 10px 14px;
    max-height: 420px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; line-height: 1.65;
}
.parse-log-box .ll-INFO  { color: #1a6b3c; }
.parse-log-box .ll-WARN  { color: #b45309; }
.parse-log-box .ll-ERROR { color: #c53030; }
.parse-log-box .ll-DEBUG { color: #94a3b8; }
/* ---- Status badge ---- */
.linked-badge {
    display:inline-block; padding:4px 10px; border-radius:4px;
    background:rgba(31,119,180,0.08); border:1px solid rgba(31,119,180,0.3);
    color:#1a365d !important; font-size:0.8rem; letter-spacing:0.06em;
}
.section-chip { background:rgba(31,119,180,0.08); color:#1a365d !important; }
"""


def apply_theme():
    """Apply the current theme CSS + set plot_utils theme.

    Call this at the top of every page (after imports) to ensure
    consistent styling regardless of which page the user is on.
    """
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'dark'

    theme = st.session_state['theme']
    plot_utils.set_theme(theme)

    theme_css = _DARK_CSS if theme == 'dark' else _LIGHT_CSS
    st.markdown(
        f"<style>{_GLOBAL_CSS}\n{theme_css}</style>",
        unsafe_allow_html=True,
    )


# ==============================================================
#  Sidebar helper utilities (shared across all pages)
# ==============================================================

def sidebar_colors() -> dict:
    """Return a theme-aware color palette for sidebar custom HTML."""
    dark = st.session_state.get('theme', 'dark') == 'dark'
    return dict(
        acc    = '#00f0ff' if dark else '#1a365d',
        txt    = '#c8d6e5' if dark else '#2d3748',
        dim    = '#8899aa' if dark else '#4a5568',
        ok     = '#64ffda' if dark else '#276749',
        warn   = '#ffd166' if dark else '#b7791f',
        err    = '#ff6b6b' if dark else '#c53030',
        bar_bg = 'rgba(255,255,255,0.08)' if dark else 'rgba(0,0,0,0.08)',
    )


def sidebar_label(text: str, c: dict = None):
    """Render a styled uppercase section label inside a sidebar block."""
    if c is None:
        c = sidebar_colors()
    st.markdown(
        f'<div style="color:{c["acc"]}; font-size:0.68rem; letter-spacing:0.12em;'
        f' font-weight:700; text-transform:uppercase; margin:6px 0 4px 0;">{text}</div>',
        unsafe_allow_html=True,
    )


def sidebar_kv(key: str, val: str, c: dict = None, val_color: str = None):
    """Render a key → value row inside a sidebar block."""
    if c is None:
        c = sidebar_colors()
    vc = val_color if val_color else c['txt']
    st.markdown(
        f'<div style="display:flex; gap:8px; font-size:0.78rem; margin:2px 0;">'
        f'<span style="color:{c["dim"]}; min-width:52px; flex-shrink:0;">{key}</span>'
        f'<span style="color:{vc};">{val}</span></div>',
        unsafe_allow_html=True,
    )


def sidebar_file_info(c: dict = None):
    """Render FILE section in sidebar if data is loaded. Returns True if rendered."""
    import data_manager as _dm
    if not _dm.is_data_loaded():
        return False
    if c is None:
        c = sidebar_colors()
    pi = _dm.get_parser_info()
    st.divider()
    sidebar_label("FILE", c)
    st.markdown(
        f'<div style="color:{c["txt"]}; font-size:0.82rem; font-weight:600;'
        f' word-break:break-all; line-height:1.4;">{pi["filename"]}</div>'
        f'<div style="color:{c["dim"]}; font-size:0.75rem; margin-top:3px;">'
        f'{pi["file_size"]/1024**2:.1f} MB &nbsp;·&nbsp; {pi["total_packets"]:,} packets</div>',
        unsafe_allow_html=True,
    )
    return True

