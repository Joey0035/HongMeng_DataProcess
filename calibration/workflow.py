"""High-level calibration workflow.

This module turns the exploratory calibration_test.ipynb flow into reusable
functions:

1. split SPEC/VNA/TEMP from a parser result,
2. apply VNA OSL calibration,
3. interpolate calibrated reflection coefficients to the SPEC frequency axis,
4. build and fit the noise-wave model,
5. recover source temperatures for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from scipy import interpolate

try:
    import skrf as rf
except ImportError:  # pragma: no cover - optional cable de-embedding dependency
    rf = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from HongMeng_DataInspector import DataInspector  # noqa: E402
from HongMeng_VNA_Calibrator import (  # noqa: E402
    CableOffset,
    Keysight85033E,
    calibrate_vna_data as _calibrate_vna_data,
)

from . import n2_generate_iter as n2  # noqa: E402


DEFAULT_SWITCH_CAL_KEYS = {
    'open': 'V_Cal_O_H',
    'short': 'V_Cal_S_H',
    'load': 'V_Cal_L_H',
}

DEFAULT_LNA_CAL_KEYS = {
    'open': 'V_LNA_O_H',
    'short': 'V_LNA_S_H',
    'load': 'V_LNA_L_H',
}


@dataclass
class CalibrationWorkflowConfig:
    """Configuration for the notebook-equivalent calibration workflow."""

    spec_channel: int = 0
    spec_freq_range_mhz: tuple[float, float] = (50.0, 180.0)
    vna_freq_hz: Optional[np.ndarray] = None
    vna_n_freq: Optional[int] = None
    trim_to_common_length: bool = True
    temp_offset_k: float = 273.15 + 10.0  # Temperature offset for probes
    ambient_temp_index: tuple[int, int] = (0, 0)
    hotload_temp_index: tuple[int, int] = (2, 0)
    receiver_source: str = 'V_LNAM_H'
    receiver_cable_snp: Optional[str] = None
    switch_cal_keys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SWITCH_CAL_KEYS))
    lna_cal_keys: Optional[dict[str, str]] = field(default_factory=lambda: dict(DEFAULT_LNA_CAL_KEYS))
    use_legacy_vna_cal: bool = False
    ns_on: str = 'NSon'
    ns_off: str = 'NSoff'
    hotload: str = 'HL'
    ambient_load: str = 'Cal_L'
    cal_sources: tuple[str, ...] = (
        'LgO', 'LgS', 'ShtS', 'ShtR1', 'ShtR2',
        'R3', 'R4', 'R5', 'Cal_O', 'Cal_S',
    )
    noise_source_terms: tuple[int, int] = (5, 5)
    noise_wave_fit_term: int = 5
    noise_wave_iterations: int = 10


@dataclass
class CalibrationWorkflowResult:
    config: CalibrationWorkflowConfig
    spec_freq_mhz: np.ndarray
    freq_mask: np.ndarray
    fit_freq_mhz: np.ndarray
    vna_freq_hz: np.ndarray
    spec_by_src: dict
    vna_by_src: dict
    model: n2.N2_nw
    vna_diagnostics: dict
    recovered_temperatures: dict[str, np.ndarray]


def _common_length(items: dict) -> int:
    lengths = [v['time'].shape[0] for v in items.values() if 'time' in v]
    if not lengths:
        return 0
    return min(lengths)


def _interp_complex(x, y, fill_x, kind='cubic') -> np.ndarray:
    func = interpolate.interp1d(x, y, kind=kind, fill_value='extrapolate')
    return np.asarray(func(fill_x))


def _ensure_2d_complex(data) -> np.ndarray:
    data = np.stack(data) if isinstance(data, list) else np.asarray(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data.astype(complex, copy=False)


def prepare_spec_by_source(
    result: dict,
    inspector: Optional[DataInspector] = None,
    channel: int = 0,
    trim_to_common_length: bool = True,
) -> tuple[np.ndarray, dict]:
    """Split SPEC data by source and select one channel.

    Returns ``(spec_freq_mhz, spec_by_src)`` where each source entry contains
    ``data`` as ``(n_time, n_freq)`` and ``time`` as float seconds.
    """
    di = inspector or DataInspector(result)
    spec_freq = di._spec_freq_mhz()
    spec_by_src = di.split_spec_time_by_src(include_time=True)
    n = _common_length(spec_by_src) if trim_to_common_length else None

    out = {}
    for src_name, item in spec_by_src.items():
        data = np.asarray(item['data'])
        time = np.asarray(item['time'], dtype=float)
        if n:
            data = data[:n]
            time = time[:n]
        out[src_name] = {
            'time': time,
            'data': data[:, channel].astype(float, copy=False),
        }
    return spec_freq, out


def prepare_vna_by_source(
    result: dict,
    vna_freq_hz: Optional[np.ndarray] = None,
    n_freq: Optional[int] = None,
    inspector: Optional[DataInspector] = None,
    trim_to_common_length: bool = True,
) -> tuple[np.ndarray, dict]:
    """Split VNA data by source.

    If ``vna_freq_hz`` is not supplied, the frequency axis is taken from
    ``DataInspector`` using ``n_freq`` or inferred from the first sweep length.
    """
    di = inspector or DataInspector(result)
    vna_by_src = di.split_vna_time_by_src(n_freq=n_freq, include_time=True)
    n = _common_length(vna_by_src) if trim_to_common_length else None

    out = {}
    inferred_n = None
    for src_name, item in vna_by_src.items():
        data = _ensure_2d_complex(item['data'])
        time = np.asarray(item['time'], dtype=float)
        if n:
            data = data[:n]
            time = time[:n]
        inferred_n = data.shape[1] if inferred_n is None else inferred_n
        out[src_name] = {'time': time, 'data': data}

    if vna_freq_hz is None:
        n_axis = n_freq or inferred_n
        freq_mhz = di._vna_freq_mhz(int(n_axis)) if n_axis is not None else None
        if freq_mhz is None:
            raise ValueError("vna_freq_hz must be supplied when n_freq is not configured")
        vna_freq_hz = np.asarray(freq_mhz, dtype=float) * 1e6
    return np.asarray(vna_freq_hz, dtype=float), out


def calibrate_vna_by_source(
    vna_by_src: dict,
    vna_freq_hz: np.ndarray,
    switch_cal_keys: Optional[dict[str, str]] = None,
    lna_cal_keys: Optional[dict[str, str]] = None,
    cable_offset: Optional[CableOffset] = None,
    use_legacy: bool = False,
) -> tuple[dict, dict]:
    """Calibrate VNA source dictionary and store ``cal_data`` per source."""
    if use_legacy:
        from .vna_cal import calibrate_vna_sources

        return calibrate_vna_sources(
            vna_by_src,
            vna_freq_hz,
            switch_cal_keys=switch_cal_keys or DEFAULT_SWITCH_CAL_KEYS,
            lna_cal_keys=lna_cal_keys or DEFAULT_LNA_CAL_KEYS,
        )

    split_data = {name: item['data'] for name, item in vna_by_src.items()}
    calibrated, diagnostics = _calibrate_vna_data(
        split_data=split_data,
        freq_hz=np.asarray(vna_freq_hz, dtype=float),
        switch_cal_keys=switch_cal_keys or DEFAULT_SWITCH_CAL_KEYS,
        lna_cal_keys=lna_cal_keys,
        cal_standard=Keysight85033E(),
        cable_offset=cable_offset,
    )

    out = {}
    for name, item in vna_by_src.items():
        new_item = dict(item)
        new_item['cal_data'] = _ensure_2d_complex(calibrated.get(name, item['data']))
        out[name] = new_item
    return out, diagnostics


def interpolate_vna_to_spec_freq(
    vna_by_src: dict,
    vna_freq_hz: np.ndarray,
    spec_freq_mhz: np.ndarray,
    freq_mask: np.ndarray,
    kind: str = 'cubic',
) -> dict:
    """Interpolate each calibrated VNA sweep to the selected SPEC frequency axis."""
    fit_freq = np.asarray(spec_freq_mhz)[freq_mask]
    vna_freq_mhz = np.asarray(vna_freq_hz, dtype=float) / 1e6
    out = {}
    for name, item in vna_by_src.items():
        cal_data = _ensure_2d_complex(item.get('cal_data', item['data']))
        interp_data = np.empty((cal_data.shape[0], fit_freq.shape[0]), dtype=complex)
        for i, sweep in enumerate(cal_data):
            interp_data[i] = _interp_complex(vna_freq_mhz, sweep, fit_freq, kind=kind)
        new_item = dict(item)
        new_item['cal_data_interpolate'] = interp_data
        out[name] = new_item
    return out


def deembed_receiver_cable(
    vna_by_src: dict,
    fit_freq_mhz: np.ndarray,
    receiver_source: str = 'V_LNAM_H',
    cable_snp: Optional[str] = None,
    kind: str = 'cubic',
) -> dict:
    """Apply scikit-rf cable de-embedding to receiver reflection data."""
    if cable_snp is None:
        return vna_by_src
    if rf is None:
        raise ImportError("scikit-rf is required for receiver cable de-embedding")
    if receiver_source not in vna_by_src:
        raise KeyError(f"receiver source not found: {receiver_source}")

    out = {k: dict(v) for k, v in vna_by_src.items()}
    cable = rf.Network(str(cable_snp)).interpolate(np.asarray(fit_freq_mhz) * 1e6, kind=kind)
    freq_obj = rf.Frequency.from_f(fit_freq_mhz, unit='MHz')
    data = _ensure_2d_complex(out[receiver_source]['cal_data_interpolate'])

    corrected = np.empty_like(data)
    for i, sweep in enumerate(data):
        nw = rf.Network(frequency=freq_obj, s=sweep.reshape(-1, 1, 1))
        corrected[i] = (cable.inv ** nw).s[:, 0, 0]
    out[receiver_source]['cal_data_interpolate'] = corrected
    return out


def build_noise_wave_model(
    fit_freq_mhz: np.ndarray,
    spec_by_src: dict,
    vna_by_src: dict,
    temp_time: np.ndarray,
    temp_data_k: np.ndarray,
    ambient_temp_index: tuple[int, int] = (0, 0),
    hotload_temp_index: tuple[int, int] = (2, 0),
    average_sources: bool = True,
) -> n2.N2_nw:
    """Create the ``N2_nw`` model used by the legacy notebook flow."""
    model = n2.N2_nw(
        fit_freq_mhz,
        spec_by_src,
        vna_by_src,
        temp_time,
        temp_data_k,
        ambient_temp_index=ambient_temp_index,
        hotload_temp_index=hotload_temp_index,
    )
    model.load_data()
    if average_sources:
        for src_name in model.src_list:
            model.src[src_name].average()
    return model


def fit_noise_wave_model(
    model: n2.N2_nw,
    config: CalibrationWorkflowConfig,
) -> n2.N2_nw:
    """Fit noise-source and receiver noise-wave parameters."""
    for src in (config.ns_on, config.ns_off):
        if src in model.src:
            model.src[src].Gsrc = np.zeros(model.freq.shape, dtype=complex)

    for src in model.src.values():
        src.get_K(src.Gsrc, src.Grec)
        src.get_X(src.P, model.src[config.ns_on].P, model.src[config.ns_off].P, src.Gsrc, src.Grec)

    tnon_term, tnoff_term = config.noise_source_terms
    model.cal_ns_temp(
        model.src[config.hotload],
        model.src[config.ambient_load],
        model.src[config.ns_on],
        model.src[config.ns_off],
        tnon_term,
        tnoff_term,
        with_nwp=False,
    )

    cal_sources = [model.src[name] for name in config.cal_sources if name in model.src]
    if not cal_sources:
        raise ValueError("No calibration sources found in the noise-wave model")

    model.Tunc, model.Tcos, model.Tsin = model.nwp_fit(
        cal_sources,
        model.TNon_poly,
        model.TNoff_poly,
        fit_term=config.noise_wave_fit_term,
    )

    for _ in range(config.noise_wave_iterations):
        model.cal_ns_temp(
            model.src[config.hotload],
            model.src[config.ambient_load],
            model.src[config.ns_on],
            model.src[config.ns_off],
            tnon_term,
            tnoff_term,
            with_nwp=True,
        )
        model.Tunc, model.Tcos, model.Tsin = model.nwp_fit(
            cal_sources,
            model.TNon_poly,
            model.TNoff_poly,
            fit_term=config.noise_wave_fit_term,
        )

    model.Tunc_fbf, model.Tcos_fbf, model.Tsin_fbf = model.nwp_fbf(
        cal_sources, 
        model.TNon_poly, 
        model.TNoff_poly,
    )

    model.cal_ns_temp(
        model.src[config.hotload],
        model.src[config.ambient_load],
        model.src[config.ns_on],
        model.src[config.ns_off],
        tnon_term,
        tnoff_term,
        with_nwp=True,
    )
    return model


def recover_source_temperatures(model: n2.N2_nw, source_names: Optional[list[str]] = None) -> dict[str, np.ndarray]:
    """Recover fitted source temperature spectra for all or selected sources."""
    names = source_names or list(model.src.keys())
    recovered = {}
    for name in names:
        src = model.src[name]
        recovered[name] = model.Tsrc_recover(
            src.Gsrc,
            src.Grec,
            src.K,
            src.P,
            src.PNS,
            src.PL,
            model.TNon_poly,
            model.Tunc_poly,
            model.Tcos_poly,
            model.Tsin_poly,
            model.TNoff_poly,
        )
        src.Ts_fit = recovered[name]
    return recovered


def run_calibration_workflow(
    result: dict,
    config: Optional[CalibrationWorkflowConfig] = None,
) -> CalibrationWorkflowResult:
    """Run the full calibration_test.ipynb workflow as a single API call."""
    cfg = config or CalibrationWorkflowConfig()
    inspector = DataInspector(result)

    spec_freq, spec_by_src = prepare_spec_by_source(
        result,
        inspector=inspector,
        channel=cfg.spec_channel,
        trim_to_common_length=cfg.trim_to_common_length,
    )
    freq_mask = (spec_freq >= cfg.spec_freq_range_mhz[0]) & (spec_freq <= cfg.spec_freq_range_mhz[1])
    spec_fit = {
        name: {'time': item['time'], 'data': item['data'][:, freq_mask]}
        for name, item in spec_by_src.items()
    }

    vna_freq_hz, vna_by_src = prepare_vna_by_source(
        result,
        vna_freq_hz=cfg.vna_freq_hz,
        n_freq=cfg.vna_n_freq,
        inspector=inspector,
        trim_to_common_length=cfg.trim_to_common_length,
    )
    vna_by_src_raw = vna_by_src
    vna_by_src, vna_diag = calibrate_vna_by_source(
        vna_by_src,
        vna_freq_hz,
        switch_cal_keys=cfg.switch_cal_keys,
        lna_cal_keys=cfg.lna_cal_keys,
        use_legacy=cfg.use_legacy_vna_cal,
    )
    vna_by_src = interpolate_vna_to_spec_freq(vna_by_src, vna_freq_hz, spec_freq, freq_mask)
    vna_by_src = deembed_receiver_cable(
        vna_by_src,
        spec_freq[freq_mask],
        receiver_source=cfg.receiver_source,
        cable_snp=cfg.receiver_cable_snp,
    )

    temp_time = np.asarray(result['temp']['time'], dtype=float)
    temp_data_k = np.asarray(result['temp']['data'], dtype=float) + cfg.temp_offset_k
    model = build_noise_wave_model(
        spec_freq[freq_mask],
        spec_fit,
        vna_by_src,
        temp_time,
        temp_data_k,
        ambient_temp_index=cfg.ambient_temp_index,
        hotload_temp_index=cfg.hotload_temp_index,
    )
    model = fit_noise_wave_model(model, cfg)
    recovered = recover_source_temperatures(model)

    return CalibrationWorkflowResult(
        config=cfg,
        spec_freq_mhz=spec_freq,
        freq_mask=freq_mask,
        fit_freq_mhz=spec_freq[freq_mask],
        vna_freq_hz=vna_freq_hz,
        spec_by_src=spec_fit,
        vna_by_src=vna_by_src,
        model=model,
        vna_diagnostics=vna_diag,
        recovered_temperatures=recovered,
    )
