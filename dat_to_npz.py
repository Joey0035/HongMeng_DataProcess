#!/usr/bin/env python3
"""dat_to_npz.py — Pre-process a .dat file into a compact .npz for fast Streamlit loading.

Designed for files too large to upload via browser (> 2 GB).
The resulting .npz can be loaded directly from the Streamlit app via the "Load NPZ" option.

Usage
-----
    python dat_to_npz.py <input.dat> [output.npz]

    If output path is omitted, the file is saved as
        {input_stem}_Parced_v3.npz  (same directory as the input).

Examples
--------
    python dat_to_npz.py /data/obs_20260401.dat
    python dat_to_npz.py /data/obs_20260401.dat /out/obs_20260401.npz
"""

import sys
import time
from pathlib import Path

# Make sure the project root is importable
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from HongMeng_raw_data_Parser import HongMengFileProcessor  # noqa: E402


def convert(dat_path: str, out_path: str | None = None) -> Path:
    dat = Path(dat_path).resolve()
    if not dat.exists():
        print(f"[ERROR] File not found: {dat}")
        sys.exit(1)

    size_gb = dat.stat().st_size / 1024 ** 3
    print(f"[INFO ] Input : {dat.name}  ({size_gb:.3f} GB)")

    t0 = time.time()
    processor = HongMengFileProcessor(verbose=True)
    # process_file with save=True writes the npz automatically
    processor.process_file(str(dat), save=True)
    elapsed = time.time() - t0

    # The parser writes to {stem}_Parced_v3.npz in the same directory
    default_out = dat.parent / f"{dat.stem}_Parced_v3.npz"

    if out_path is not None:
        target = Path(out_path).resolve()
        if target != default_out:
            target.parent.mkdir(parents=True, exist_ok=True)
            default_out.rename(target)
            final = target
        else:
            final = default_out
    else:
        final = default_out

    size_npz_mb = final.stat().st_size / 1024 ** 2
    print(f"[INFO ] Output: {final}  ({size_npz_mb:.1f} MB)")
    print(f"[INFO ] Done in {elapsed:.1f}s")
    return final


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    convert(
        dat_path=sys.argv[1],
        out_path=sys.argv[2] if len(sys.argv) > 2 else None,
    )
