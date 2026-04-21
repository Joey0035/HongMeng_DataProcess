"""
config.py — Constants, source labels, and default values.
"""

# ==============================================================
#  Source label map (from HongMeng_Preprocessor.py sw_def)
# ==============================================================

SOURCE_LABEL_MAP = {
    0: 'V_Ant_H',    1: 'V_NSon_H',   2: 'V_NSoff_H',  3: 'V_HL_H',
    4: 'V_LgO_H',    5: 'V_LgS_H',    6: 'V_Cal_L_H',  7: 'V_Cal_O_H',
    8: 'V_Cal_S_H',  9: 'V_R3_H',    10: 'V_R4_H',    11: 'V_R5_H',
   12: 'V_ShtO_H',  13: 'V_ShtS_H',  14: 'V_ShtR1_H', 15: 'V_ShtR2_H',
   20: 'V_LNAM_H',  21: 'V_LNA_O_H', 22: 'V_LNA_S_H', 23: 'V_LNA_L_H',
   30: 'Ant_H',     31: 'NSon_H',    32: 'NSoff_H',   33: 'HL_H',
   34: 'LgO_H',     35: 'LgS_H',     36: 'Cal_L_H',   37: 'Cal_O_H',
   38: 'Cal_S_H',   39: 'R3_H',      40: 'R4_H',      41: 'R5_H',
   42: 'ShtO_H',    43: 'ShtS_H',    44: 'ShtR1_H',   45: 'ShtR2_H',
}

# ==============================================================
#  SPEC constants
# ==============================================================

SPEC_BW_MHZ = 250.0
SPEC_N_CH = 4096
PKTS_PER_FFT = 64

DTYPE_INDEX_MAP = {
    'Auto1': 0,
    'Auto2': 1,
    'Cross-Imag': 2,
    'Cross-Real': 3,
}

DTYPE_LABELS = list(DTYPE_INDEX_MAP.keys())

# ==============================================================
#  VNA defaults
# ==============================================================

DEFAULT_VNA_FREQ = {901: (30, 120), 1901: (1, 190)}

# ==============================================================
#  Temperature
# ==============================================================

TEMP_N_CHIPS = 5
TEMP_CH_PER_CHIP = 5

TEMP_SENSOR_LABELS = [
    f"Chip{i}-Ch{j}" for i in range(TEMP_N_CHIPS) for j in range(TEMP_CH_PER_CHIP)
]

DEFAULT_TEMP_THRESHOLDS = {'high': 60.0, 'low': -10.0}

# ==============================================================
#  Anomaly display names
# ==============================================================

ANOMALY_CATEGORIES = {
    'timestamp': 'Timestamp',
    'seq_gap': 'Seq Gap',
    'vna_sweep': 'VNA Sweep',
    'temp_nan': 'TEMP NaN',
}

# ==============================================================
#  Plot display limits (waterfall downsampling)
# ==============================================================

MAX_WATERFALL_ROWS = 500    # max time steps in waterfall heatmap
MAX_1D_POINTS = 4096        # max freq points per 1D trace (no effect for SPEC)

# ==============================================================
#  VNA Calibration
# ==============================================================

# 12-port switch plane calibration standard source keys
CAL_SWITCH_PLANE = {
    'open':  'V_Cal_O_H',   # src 7
    'short': 'V_Cal_S_H',   # src 8
    'load':  'V_Cal_L_H',   # src 6
}

# LNA plane calibration standard source keys
CAL_LNA_PLANE = {
    'open':  'V_LNA_O_H',   # src 21
    'short': 'V_LNA_S_H',   # src 22
    'load':  'V_LNA_L_H',   # src 23
}

# LNA plane cable offset defaults (6cm RF cable)
CAL_LNA_CABLE_DEFAULTS = {
    'length_m': 0.06,
    'velocity_factor': 0.695,
    'loss_db_per_m_per_ghz': 0.0,
}

# ==============================================================
#  Streaming parser thresholds
# ==============================================================

STREAM_THRESHOLD = 512 * 1024 * 1024    # 512 MB, above this use streaming
CHUNK_SIZE = 64 * 1024 * 1024           # 64 MB per read chunk
