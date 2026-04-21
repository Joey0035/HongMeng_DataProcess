"""
HongMeng_VNA_Calibrator.py — VNA OSL (Open-Short-Load) calibration engine.

Implements 1-port 3-term error model for Keysight 85033E calibration kit.
Two calibration planes:
  - 12-port switch plane (V_Cal_O/S/L_H)
  - LNA plane (V_LNA_O/S/L_H) with cable de-embedding

Pure numpy — no Streamlit or skrf dependency.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================
#  Keysight 85033E calibration standard coefficients
# ==============================================================

KEYSIGHT_85033E = {
    'open': {
        'C0': 49.433e-15,       # F
        'C1': -310.13e-27,      # F/Hz
        'C2': 23.168e-36,       # F/Hz^2
        'C3': -0.15966e-45,     # F/Hz^3
        'offset_delay': 29.243e-12,  # s (one-way)
        'offset_loss': 2.2e9,        # Ω/s (loss term, GΩ/s)
        'offset_z0': 50.0,           # Ω
    },
    'short': {
        'L0': 2.0765e-12,       # H
        'L1': -108.54e-24,      # H/Hz
        'L2': 2.1705e-33,       # H/Hz^2
        'L3': -0.01e-42,        # H/Hz^3
        'offset_delay': 31.785e-12,  # s (one-way)
        'offset_loss': 2.36e9,       # Ω/s
        'offset_z0': 50.0,           # Ω
    },
    'load': {
        'Z': 50.0,  # Ω (ideal match)
    },
}

C_LIGHT = 299792458.0  # m/s


# ==============================================================
#  Keysight 85033E standard model
# ==============================================================

class Keysight85033E:
    """Compute the known reflection coefficients of 85033E standards."""

    def __init__(self, coeffs: dict | None = None, z0: float = 50.0):
        self.coeffs = coeffs or KEYSIGHT_85033E
        self.z0 = z0

    def gamma_open(self, freq_hz: np.ndarray) -> np.ndarray:
        """Known Γ of the Open standard vs frequency."""
        c = self.coeffs['open']
        f = freq_hz.astype(np.float64)
        omega = 2.0 * np.pi * f

        # Fringing capacitance model
        C = c['C0'] + c['C1'] * f + c['C2'] * f**2 + c['C3'] * f**3

        # Reflection from capacitive termination
        jωCZ0 = 1j * omega * C * self.z0
        gamma_cap = (1.0 - jωCZ0) / (1.0 + jωCZ0)

        # Offset delay (round-trip phase)
        delay = c['offset_delay']
        gamma = gamma_cap * np.exp(-1j * 2.0 * omega * delay)

        return gamma

    def gamma_short(self, freq_hz: np.ndarray) -> np.ndarray:
        """Known Γ of the Short standard vs frequency."""
        c = self.coeffs['short']
        f = freq_hz.astype(np.float64)
        omega = 2.0 * np.pi * f

        # Inductance model
        L = c['L0'] + c['L1'] * f + c['L2'] * f**2 + c['L3'] * f**3

        # Reflection from inductive termination
        jωL = 1j * omega * L
        gamma_ind = (jωL - self.z0) / (jωL + self.z0)

        # Offset delay (round-trip phase)
        delay = c['offset_delay']
        gamma = gamma_ind * np.exp(-1j * 2.0 * omega * delay)

        return gamma

    def gamma_load(self, freq_hz: np.ndarray) -> np.ndarray:
        """Known Γ of the Load standard (ideal 50Ω match)."""
        return np.zeros(len(freq_hz), dtype=np.complex128)


# ==============================================================
#  Minimal Touchstone (.s1p / .s2p) parser
# ==============================================================

def read_touchstone(filepath: str) -> dict:
    """Parse a Touchstone .s1p or .s2p file.

    Returns
    -------
    dict with keys:
        'freq_hz' : np.ndarray (n_freq,)
        'data'    : np.ndarray — .s1p: (n_freq,) complex128
                                  .s2p: (n_freq, 2, 2) complex128
        'z0'      : float
        'n_ports' : int (1 or 2)
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == '.s1p':
        n_ports = 1
    elif suffix == '.s2p':
        n_ports = 2
    else:
        raise ValueError(f"Unsupported Touchstone extension: {suffix}")

    freq_mult = 1e9  # default GHz
    fmt = 'ma'       # default magnitude-angle
    z0 = 50.0
    param = 's'

    freq_list = []
    data_rows = []

    with open(filepath, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            if line.startswith('#'):
                # Option line: # <freq_unit> <param> <format> R <z0>
                tokens = line[1:].split()
                for i, tok in enumerate(tokens):
                    tok_l = tok.lower()
                    if tok_l in ('hz',):
                        freq_mult = 1.0
                    elif tok_l in ('khz',):
                        freq_mult = 1e3
                    elif tok_l in ('mhz',):
                        freq_mult = 1e6
                    elif tok_l in ('ghz',):
                        freq_mult = 1e9
                    elif tok_l in ('s', 'y', 'z', 'h', 'g'):
                        param = tok_l
                    elif tok_l in ('ma', 'db', 'ri'):
                        fmt = tok_l
                    elif tok_l == 'r':
                        if i + 1 < len(tokens):
                            z0 = float(tokens[i + 1])
                continue

            # Data line
            vals = [float(x) for x in line.split()]
            if not vals:
                continue
            freq_list.append(vals[0])
            data_rows.append(vals[1:])

    freq_hz = np.array(freq_list) * freq_mult
    raw = np.array(data_rows)

    def _to_complex(col1: np.ndarray, col2: np.ndarray) -> np.ndarray:
        if fmt == 'ri':
            return col1 + 1j * col2
        elif fmt == 'ma':
            return col1 * np.exp(1j * np.deg2rad(col2))
        elif fmt == 'db':
            mag = 10.0 ** (col1 / 20.0)
            return mag * np.exp(1j * np.deg2rad(col2))
        raise ValueError(f"Unknown format: {fmt}")

    if n_ports == 1:
        data = _to_complex(raw[:, 0], raw[:, 1])
    else:
        # .s2p: 8 columns → S11, S21, S12, S22 (each 2 values)
        s11 = _to_complex(raw[:, 0], raw[:, 1])
        s21 = _to_complex(raw[:, 2], raw[:, 3])
        s12 = _to_complex(raw[:, 4], raw[:, 5])
        s22 = _to_complex(raw[:, 6], raw[:, 7])
        data = np.zeros((len(freq_hz), 2, 2), dtype=np.complex128)
        data[:, 0, 0] = s11
        data[:, 0, 1] = s12
        data[:, 1, 0] = s21
        data[:, 1, 1] = s22

    return {'freq_hz': freq_hz, 'data': data, 'z0': z0, 'n_ports': n_ports}


# ==============================================================
#  Cable offset model (analytical + SNP override)
# ==============================================================

class CableOffset:
    """Model an RF cable for calibration plane de-embedding.

    Analytical model: lossless or lossy transmission line.
    SNP override: load measured S-parameters from .s1p/.s2p file.
    """

    def __init__(
        self,
        length_m: float = 0.06,
        velocity_factor: float = 0.695,
        loss_db_per_m_per_ghz: float = 0.0,
    ):
        self.length_m = length_m
        self.velocity_factor = velocity_factor
        self.loss_db_per_m_per_ghz = loss_db_per_m_per_ghz
        # SNP override data (set by from_snp_file)
        self._snp_freq_hz: np.ndarray | None = None
        self._snp_s21: np.ndarray | None = None
        self._snp_s11: np.ndarray | None = None
        self._snp_s22: np.ndarray | None = None

    @classmethod
    def from_snp_file(cls, filepath: str, **analytical_defaults) -> 'CableOffset':
        """Create a CableOffset using measured S-parameters from SNP file."""
        obj = cls(**analytical_defaults)
        snp = read_touchstone(filepath)
        obj._snp_freq_hz = snp['freq_hz']
        if snp['n_ports'] == 2:
            obj._snp_s11 = snp['data'][:, 0, 0]
            obj._snp_s21 = snp['data'][:, 1, 0]
            obj._snp_s22 = snp['data'][:, 1, 1]
        elif snp['n_ports'] == 1:
            # .s1p only has S11 of the cable — treat as reflection-only
            obj._snp_s11 = snp['data']
            obj._snp_s21 = None
            obj._snp_s22 = None
        return obj

    def _analytical_s21(self, freq_hz: np.ndarray) -> np.ndarray:
        """S21 of an ideal transmission line."""
        f = freq_hz.astype(np.float64)
        v_phase = C_LIGHT * self.velocity_factor
        beta = 2.0 * np.pi * f / v_phase
        phase = beta * self.length_m

        if self.loss_db_per_m_per_ghz > 0:
            alpha_neper = (self.loss_db_per_m_per_ghz * self.length_m
                           * f / 1e9 / 20.0 * np.log(10))
            return np.exp(-alpha_neper) * np.exp(-1j * phase)
        return np.exp(-1j * phase)

    def _interp_snp(self, freq_hz: np.ndarray, snp_data: np.ndarray) -> np.ndarray:
        """Interpolate SNP data to target frequencies."""
        mag = np.abs(snp_data)
        phase = np.unwrap(np.angle(snp_data))
        mag_interp = np.interp(freq_hz, self._snp_freq_hz, mag)
        phase_interp = np.interp(freq_hz, self._snp_freq_hz, phase)
        return mag_interp * np.exp(1j * phase_interp)

    def deembed(self, gamma_measured: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
        """De-embed the cable from measured reflection coefficient.

        For a 2-port cable [S11, S12; S21, S22]:
            Γ_DUT = (Γ_meas - S11) / (S22 * (Γ_meas - S11) + S12 * S21)

        For an ideal matched cable (S11=S22=0):
            Γ_DUT = Γ_meas / S21^2  (remove round-trip through cable)
        """
        if self._snp_freq_hz is not None and self._snp_s21 is not None:
            # Full 2-port SNP de-embedding
            s11 = self._interp_snp(freq_hz, self._snp_s11)
            s21 = self._interp_snp(freq_hz, self._snp_s21)
            s22 = self._interp_snp(freq_hz, self._snp_s22)
            s12 = s21  # assume reciprocal
            diff = gamma_measured - s11
            return diff / (s22 * diff + s12 * s21)

        # Analytical model
        s21 = self._analytical_s21(freq_hz)
        return gamma_measured / (s21 ** 2)


# ==============================================================
#  1-port 3-term error model solver
# ==============================================================

def solve_osl_error_terms(
    s11m_open: np.ndarray,
    s11m_short: np.ndarray,
    s11m_load: np.ndarray,
    gamma_open: np.ndarray,
    gamma_short: np.ndarray,
    gamma_load: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 1-port 3-term error model from OSL measurements.

    Error model: S11m = (a + b·Γ) / (1 - c·Γ)
    where a = e_d (directivity), c = e_s (source match),
    and e_r = a·c + b (reflection tracking).

    Rearranged as a linear system in (a, b, c):
        a + b·Γ_x + c·S11m_x·Γ_x = S11m_x   for x ∈ {O, S, L}

    Parameters
    ----------
    s11m_open, s11m_short, s11m_load : (n_freq,) complex
        Measured S11 of each standard.
    gamma_open, gamma_short, gamma_load : (n_freq,) complex
        Known Γ of each standard from the calibration kit model.

    Returns
    -------
    (e_d, e_s, e_r) : each (n_freq,) complex128
    """
    n = len(s11m_open)

    # Build (n_freq, 3, 3) coefficient matrix and (n_freq, 3) RHS
    A = np.zeros((n, 3, 3), dtype=np.complex128)
    b = np.zeros((n, 3), dtype=np.complex128)

    # Row 0: Open
    A[:, 0, 0] = 1.0
    A[:, 0, 1] = gamma_open
    A[:, 0, 2] = s11m_open * gamma_open
    b[:, 0] = s11m_open

    # Row 1: Short
    A[:, 1, 0] = 1.0
    A[:, 1, 1] = gamma_short
    A[:, 1, 2] = s11m_short * gamma_short
    b[:, 1] = s11m_short

    # Row 2: Load
    A[:, 2, 0] = 1.0
    A[:, 2, 1] = gamma_load
    A[:, 2, 2] = s11m_load * gamma_load
    b[:, 2] = s11m_load

    # Solve vectorized: A is (n, 3, 3), b is (n, 3) → need (n, 3, 1) for batched solve
    try:
        x = np.linalg.solve(A, b[:, :, np.newaxis]).squeeze(-1)  # (n_freq, 3)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix in OSL solver — falling through to per-freq solve")
        x = np.zeros((n, 3), dtype=np.complex128)
        for i in range(n):
            try:
                x[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                x[i] = [0, 0, 0]

    a = x[:, 0]  # e_d (directivity)
    b_val = x[:, 1]
    c = x[:, 2]  # e_s (source match)

    e_d = a
    e_s = c
    e_r = a * c + b_val  # reflection tracking

    # Condition number check
    cond = np.linalg.cond(A)
    max_cond = np.max(cond)
    if max_cond > 1e10:
        logger.warning(f"OSL solver: max condition number = {max_cond:.2e} — results may be unreliable")

    return e_d, e_s, e_r


def apply_osl_correction(
    s11_measured: np.ndarray,
    e_d: np.ndarray,
    e_s: np.ndarray,
    e_r: np.ndarray,
) -> np.ndarray:
    """Apply 1-port error correction.

    S11_corr = (S11m - e_d) / (e_s * (S11m - e_d) + e_r)

    Parameters
    ----------
    s11_measured : (n_freq,) or (n_sweep, n_freq) complex
    e_d, e_s, e_r : (n_freq,) complex

    Returns
    -------
    Corrected S11, same shape as input.
    """
    if s11_measured.ndim == 2:
        # Broadcast: (n_sweep, n_freq) vs (n_freq,)
        diff = s11_measured - e_d[np.newaxis, :]
        return diff / (e_s[np.newaxis, :] * diff + e_r[np.newaxis, :])

    diff = s11_measured - e_d
    return diff / (e_s * diff + e_r)


# ==============================================================
#  High-level calibration orchestrator
# ==============================================================

# All 6 calibration standard source names (2 planes × 3 standards)
CAL_STANDARD_NAMES = {
    'V_Cal_O_H', 'V_Cal_S_H', 'V_Cal_L_H',
    'V_LNA_O_H', 'V_LNA_S_H', 'V_LNA_L_H',
}


def _is_lna_source(name: str) -> bool:
    """Check if a source should be calibrated via the LNA plane.

    Only non-standard sources containing 'LNA' go through LNA plane.
    """
    return 'LNA' in name and name not in CAL_STANDARD_NAMES


def _ensure_2d(data) -> np.ndarray:
    """Convert list or 1D to 2D (n_sweep, n_freq)."""
    if isinstance(data, list):
        data = np.stack(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def _get_standard_data(split_data: dict, key: str) -> np.ndarray | None:
    """Get standard data as 2D array (n_sweep, n_freq), or None if missing."""
    if key not in split_data:
        return None
    return _ensure_2d(split_data[key])


def calibrate_vna_data(
    split_data: dict,
    freq_hz: np.ndarray,
    switch_cal_keys: dict | None = None,
    lna_cal_keys: dict | None = None,
    cal_standard: Keysight85033E | None = None,
    cable_offset: CableOffset | None = None,
) -> tuple[dict, dict]:
    """Full VNA OSL calibration pipeline — per-cycle independent calibration.

    Each observation cycle is calibrated independently using that cycle's
    own standard measurements. This avoids averaging-out temporal drift.

    Parameters
    ----------
    split_data : dict
        {src_name: (n_sweep, n_freq) complex128} from split_vna_by_src().
        Each source has n_sweep sweeps, where sweep i corresponds to cycle i.
    freq_hz : np.ndarray
        (n_freq,) frequency axis in Hz.
    switch_cal_keys : dict
        {'open': 'V_Cal_O_H', 'short': 'V_Cal_S_H', 'load': 'V_Cal_L_H'}
    lna_cal_keys : dict or None
        {'open': 'V_LNA_O_H', 'short': 'V_LNA_S_H', 'load': 'V_LNA_L_H'}
    cal_standard : Keysight85033E or None
        Standard model (default: 85033E with default coefficients).
    cable_offset : CableOffset or None
        Cable de-embedding for LNA plane.

    Returns
    -------
    (calibrated_data, diagnostics)
        calibrated_data: dict {src_name: (n_sweep, n_freq) complex128}
            Includes both DUT sources and standards (standards self-calibrated).
        diagnostics: dict with error terms, standard self-check, and metadata.
    """
    if cal_standard is None:
        cal_standard = Keysight85033E()

    if switch_cal_keys is None:
        switch_cal_keys = {
            'open': 'V_Cal_O_H', 'short': 'V_Cal_S_H', 'load': 'V_Cal_L_H',
        }

    diagnostics = {
        'switch_plane': None,
        'lna_plane': None,
        'calibrated_sources': [],
        'uncalibrated_sources': [],
        'standards_self_cal': {},   # {std_name: (n_cycle, n_freq) complex}
    }

    # ---- Known Gamma from standard model ----
    gamma_o = cal_standard.gamma_open(freq_hz)
    gamma_s = cal_standard.gamma_short(freq_hz)
    gamma_l = cal_standard.gamma_load(freq_hz)

    # ---- Retrieve standard arrays ----
    sw_open = _get_standard_data(split_data, switch_cal_keys['open'])
    sw_short = _get_standard_data(split_data, switch_cal_keys['short'])
    sw_load = _get_standard_data(split_data, switch_cal_keys['load'])

    sw_available = (sw_open is not None and sw_short is not None and sw_load is not None)
    if sw_available:
        n_cycles_sw = min(sw_open.shape[0], sw_short.shape[0], sw_load.shape[0])
        logger.info(f"Switch-plane: {n_cycles_sw} calibration cycle(s)")
    else:
        n_cycles_sw = 0
        missing = [k for k, v in [('open', sw_open), ('short', sw_short), ('load', sw_load)] if v is None]
        logger.warning(f"Switch-plane calibration skipped — missing standards: {missing}")

    lna_open = lna_short = lna_load = None
    lna_available = False
    n_cycles_lna = 0
    if lna_cal_keys is not None:
        lna_open = _get_standard_data(split_data, lna_cal_keys['open'])
        lna_short = _get_standard_data(split_data, lna_cal_keys['short'])
        lna_load = _get_standard_data(split_data, lna_cal_keys['load'])
        lna_available = (lna_open is not None and lna_short is not None and lna_load is not None)
        if lna_available:
            n_cycles_lna = min(lna_open.shape[0], lna_short.shape[0], lna_load.shape[0])
            logger.info(f"LNA-plane: {n_cycles_lna} calibration cycle(s)")
        else:
            missing = [k for k, v in [('open', lna_open), ('short', lna_short), ('load', lna_load)] if v is None]
            logger.warning(f"LNA-plane calibration skipped — missing standards: {missing}")

    # ---- Per-cycle error terms ----
    # Compute error terms for each cycle independently
    # sw_error_terms[i] = (e_d, e_s, e_r) each (n_freq,)
    sw_error_list = []
    if sw_available:
        for i in range(n_cycles_sw):
            e_d, e_s, e_r = solve_osl_error_terms(
                sw_open[i], sw_short[i], sw_load[i],
                gamma_o, gamma_s, gamma_l,
            )
            sw_error_list.append((e_d, e_s, e_r))
        diagnostics['switch_plane'] = {
            'n_cycles': n_cycles_sw,
            'error_terms': sw_error_list,
        }

    lna_error_list = []
    if lna_available:
        for i in range(n_cycles_lna):
            e_d, e_s, e_r = solve_osl_error_terms(
                lna_open[i], lna_short[i], lna_load[i],
                gamma_o, gamma_s, gamma_l,
            )
            lna_error_list.append((e_d, e_s, e_r))
        diagnostics['lna_plane'] = {
            'n_cycles': n_cycles_lna,
            'error_terms': lna_error_list,
        }

    # ---- Helper: apply per-cycle correction to a source ----
    def _calibrate_source(data_2d: np.ndarray, error_list: list,
                          do_deembed: bool = False) -> np.ndarray:
        """Apply per-cycle error correction.

        If data has more sweeps than error cycles, cycles are reused cyclically.
        """
        n_sw = data_2d.shape[0]
        n_cyc = len(error_list)
        corrected = np.empty_like(data_2d)
        for i in range(n_sw):
            cyc = i % n_cyc  # match sweep i to cycle i (cyclic if uneven)
            e_d, e_s, e_r = error_list[cyc]
            corrected[i] = apply_osl_correction(data_2d[i], e_d, e_s, e_r)
            if do_deembed and cable_offset is not None:
                corrected[i] = cable_offset.deembed(corrected[i], freq_hz)
        return corrected

    # ---- Standards self-calibration (quality check) ----
    if sw_available:
        for key_type, std_data, gamma_known in [
            ('open', sw_open, gamma_o),
            ('short', sw_short, gamma_s),
            ('load', sw_load, gamma_l),
        ]:
            std_name = switch_cal_keys[key_type]
            n = min(std_data.shape[0], n_cycles_sw)
            cal_std = _calibrate_source(std_data[:n], sw_error_list)
            diagnostics['standards_self_cal'][std_name] = {
                'calibrated': cal_std,
                'expected': gamma_known,
            }

    if lna_available:
        for key_type, std_data, gamma_known in [
            ('open', lna_open, gamma_o),
            ('short', lna_short, gamma_s),
            ('load', lna_load, gamma_l),
        ]:
            std_name = lna_cal_keys[key_type]
            n = min(std_data.shape[0], n_cycles_lna)
            cal_std = _calibrate_source(std_data[:n], lna_error_list)
            diagnostics['standards_self_cal'][std_name] = {
                'calibrated': cal_std,
                'expected': gamma_known,
            }

    # ---- Apply correction to all sources (including standards) ----
    calibrated = {}
    for name, data in split_data.items():
        data = _ensure_2d(data)

        is_cal_standard = name in CAL_STANDARD_NAMES
        is_lna = _is_lna_source(name)

        if is_cal_standard:
            # Standards: include self-calibrated version in output
            self_cal = diagnostics['standards_self_cal'].get(name)
            if self_cal is not None:
                calibrated[name] = self_cal['calibrated']
            else:
                calibrated[name] = data
            continue

        if is_lna and lna_available:
            corrected = _calibrate_source(data, lna_error_list, do_deembed=True)
            calibrated[name] = corrected
            diagnostics['calibrated_sources'].append(name)
        elif not is_lna and sw_available:
            corrected = _calibrate_source(data, sw_error_list, do_deembed=False)
            calibrated[name] = corrected
            diagnostics['calibrated_sources'].append(name)
        else:
            calibrated[name] = data
            diagnostics['uncalibrated_sources'].append(name)

    return calibrated, diagnostics
