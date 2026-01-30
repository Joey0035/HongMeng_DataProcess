# Copilot instructions (HongMeng Data Process Tools)

## Big picture / data flow
- Raw DSL .dat files are parsed into NumPy .npz via [HongMeng_raw_data_Parser.py](HongMeng_raw_data_Parser.py). The public entrypoint is `run_ParceSpecPacket()` or the CLI in the same file.
- Parsed output is saved as `<stem>_Parced_v3.npz` with keys `time`, `sci_data`, `seq_count`, etc. (see `DSLFileProcessor._save_result`).
- Preprocess/analysis happens in [HongMeng_preprocess.py](HongMeng_preprocess.py) and the notebooks (e.g. [DSLpreprocess.ipynb](DSLpreprocess.ipynb)).

## Core data shapes and conventions
- One spectrum (spec) = 64 packets (PACKETS_PER_SPEC), 4 channels, 4096 values per channel. `sci_data` shape is `(N_specs, 4, 4096)`.
- `time` in parsed files is per-packet; when aligning with specs, code uses `time[::64]` to timestamp each spec (see `data_separation`).
- Source switching definitions (`sw_def`) map integer IDs to labels like `Ant_H`, `NSon_H`; source sequences are defined in `src_squence()` and expanded by `int_times` (default 4).

## Parsing pipeline details (performance-sensitive)
- Parsing chooses streaming vs all-in-memory based on file size (`stream_threshold` default 512 MB). Streaming uses chunked reads and sync-word scanning.
- Alignment step drops trailing packets to keep full FFT blocks (`PACKETS_PER_SPEC`). `skip_pkt` can offset initial packets for alignment.

## Calibration / utility patterns
- Frequency/time helper utilities live in [Calibration_tools.py](Calibration_tools.py). These are plain functions; avoid reimplementing similar helpers elsewhere.
- `rebin` is imported in multiple files but is not in this repo; assume an external/local module and avoid changing its call sites unless necessary.

## Developer workflows (observed)
- No test harness present; primary usage is running scripts or notebooks.
- CLI: `python HongMeng_raw_data_Parser.py <file_path> [skip_pkt] [save]` parses and optionally saves .npz.
- For preprocessing, instantiate `DSLpreprocess`/`DSLpreprocesser` with a parsed `.npz` and call `data_separation()` (see [HongMeng_preprocess.py](HongMeng_preprocess.py)).

## Project-specific guidance
- Preserve existing file naming conventions for outputs (`*_Parced_v3.npz`, `*_toCal.npz` in notebooks).
- Keep `sci_data` as int64/NumPy arrays; avoid Python loops where vectorized code already exists.
- When touching parsing, maintain `SYNC_WORD` framing and checksum logic in [HongMeng_raw_data_Parser.py](HongMeng_raw_data_Parser.py).
