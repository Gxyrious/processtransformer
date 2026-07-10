"""Batch-convert XES event logs to flat CSV, preserving every attribute.

Each ``.xes`` / ``.xes.gz`` file is read with pm4py and written next to the
source as a ``.csv``. pm4py's XES reader flattens the log so that:

* event-level attributes keep their original keys (``concept:name``,
  ``time:timestamp``, ``org:resource``, ``lifecycle:transition`` ...), and
* trace-level attributes are prefixed with ``case:`` (``case:concept:name``,
  ``case:Amount`` ...),

so nothing from the source log is dropped. Each output row is one event.

Usage::

    python xes_to_csv.py                       # convert everything under datasets/raw
    python xes_to_csv.py --input path/to/log.xes.gz
    python xes_to_csv.py --input path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pm4py


def _resolve_inputs(input_path: Path) -> list[Path]:
    """Return all .xes / .xes.gz files at or below ``input_path``."""
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*") if p.suffixes[-2:] == [".xes", ".gz"] or p.suffix == ".xes"
    )


def _output_path(xes_path: Path) -> Path:
    """``foo.xes.gz`` / ``foo.xes`` -> ``foo.csv`` in the same directory."""
    name = xes_path.name
    for ext in (".xes.gz", ".xes"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return xes_path.with_name(name + ".csv")


def convert(xes_path: Path) -> tuple[Path, pd.DataFrame]:
    """Read one XES log into a DataFrame and write it as CSV."""
    df = pm4py.read_xes(str(xes_path))
    out_path = _output_path(xes_path)
    df.to_csv(out_path, index=False)
    return out_path, df


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert XES logs to CSV (full attribute retention).")
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "datasets" / "raw",
        help="a .xes(.gz) file or a directory to scan recursively",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")

    files = _resolve_inputs(args.input)
    if not files:
        print(f"no XES files under {args.input}", file=sys.stderr)
        return 1

    failures = 0
    for xes_path in files:
        try:
            out_path, df = convert(xes_path)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures += 1
            print(f"[FAIL] {xes_path} -> {exc}", file=sys.stderr)
            continue
        print(
            f"[OK]   {xes_path.name}\n"
            f"       -> {out_path}  ({len(df)} events, {df.shape[1]} cols, "
            f"{df['case:concept:name'].nunique() if 'case:concept:name' in df else '?'} cases)"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
