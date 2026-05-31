"""Calibration utilities for HongMeng data.

The package keeps the legacy modules (`tools`, `vna_cal`, `n2_generate_iter`)
import-compatible while exposing a higher-level workflow API that mirrors the
calibration_test.ipynb process.
"""

from .workflow import (
    CalibrationWorkflowConfig,
    CalibrationWorkflowResult,
    prepare_spec_by_source,
    prepare_vna_by_source,
    calibrate_vna_by_source,
    interpolate_vna_to_spec_freq,
    deembed_receiver_cable,
    build_noise_wave_model,
    fit_noise_wave_model,
    recover_source_temperatures,
    run_calibration_workflow,
)

__all__ = [
    'CalibrationWorkflowConfig',
    'CalibrationWorkflowResult',
    'prepare_spec_by_source',
    'prepare_vna_by_source',
    'calibrate_vna_by_source',
    'interpolate_vna_to_spec_freq',
    'deembed_receiver_cable',
    'build_noise_wave_model',
    'fit_noise_wave_model',
    'recover_source_temperatures',
    'run_calibration_workflow',
]
