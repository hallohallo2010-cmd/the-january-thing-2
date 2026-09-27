#!/usr/bin/env python3
"""Append a run_forward.py cut file into the cumulative prediction ledger.

run_forward.py writes one file per cut date (data/live_predictions_<cut>.csv)
and refuses to overwrite one that exists. That is the right guarantee for a
single run, but it leaves the forward test spread across a growing pile of
files. This script folds each cut file into one append-only ledger,
data/live_predictions.csv, so the whole forward test is one artifact whose git
history is the evidence.

ROWS ARE COPIED AS BYTES, NEVER RE-SERIALIZED
---------------------------------------------
New rows are appended as their original text from the cut file. They are never
parsed into floats and written back out: pandas' default CSV float formatting
does not round-trip float64, and it silently shortens values --
-0.036036036036036063 comes back as -0.036036036036036. A recorded feature
value that drifts when it is copied is a corrupted prediction, so the numbers
this script moves are the bytes run_forward.py wrote. pandas is used only to
decide WHICH rows are new.

APPEND-ONLY, ENFORCED RATHER THAN INTENDED
------------------------------------------
Existing rows are never edited, reordered or dropped. The ledger's prior
content is captured before the write and verified as an exact byte prefix
afterwards; a mismatch aborts non-zero. A prediction that could be revised
after the fact is not evidence, so this is checked, not assumed.

THE DEDUPLICATION KEY IS (ticker, last_known_period_end)
--------------------------------------------------------
A row predicts the quarter FOLLOWING last_known_period_end -- the last quarter
the ticker had actually filed as of the cut. Two runs that see the same
last_known_period_end for a ticker are predicting the same real quarter, so the
second is a duplicate and is dropped. When the company files, its
last_known_period_end advances and the next quarter becomes a legitimately new
prediction.

expected_period_end is deliberately NOT the key. run_forward.py documents it as
an estimate -- the ticker's median quarter length added to last_known_period_end
-- so it can drift by a day between runs with no filing in between. Keying on it
would admit a second prediction for a quarter that already has one, which is
exactly what the ledger exists to prevent.

Run:  python scripts/append_predictions.py data/live_predictions_2026-10-20.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

LEDGER = "data/live_predictions.csv"
KEY = ["ticker", "last_known_period_end"]


def stop(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def split_csv(path: str) -> tuple[str, list[str]]:
    """Return (header_line, data_lines) with the `#` provenance block dropped.

    Line-level rather than field-level on purpose -- see the module docstring.
    """
    with open(path, newline="") as handle:
        lines = handle.read().splitlines()

    rows = [line for line in lines if not line.startswith("#") and line.strip()]
    if not rows:
        stop(f"{path} has no CSV rows.")
    return rows[0], rows[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cut_file", help="the per-cut CSV run_forward.py wrote")
    args = parser.parse_args()

    if not os.path.exists(args.cut_file):
        stop(f"{args.cut_file} not found.")

    header, data_lines = split_csv(args.cut_file)
    frame = pd.read_csv(args.cut_file, comment="#", dtype=str)

    if len(frame) != len(data_lines):
        # The raw lines are matched to the parsed rows positionally, so any
        # disagreement (an embedded newline, a stray blank) means the mapping
        # is unsafe and the wrong bytes could be appended.
        stop(f"{args.cut_file}: {len(frame)} parsed rows but {len(data_lines)} "
             f"raw lines. Refusing to append on an ambiguous row mapping.")

    for column in KEY:
        if column not in frame.columns:
            stop(f"{args.cut_file} has no {column} column; the ledger cannot be "
                 f"deduplicated without it.")

    # Captured before the write so the post-write check compares against the
    # real prior bytes, not a re-serialization of them.
    previous_bytes = b""
    already: set[tuple[str, ...]] = set()

    if os.path.exists(LEDGER):
        with open(LEDGER, "rb") as handle:
            previous_bytes = handle.read()
        ledger_header, _ = split_csv(LEDGER)
        if ledger_header != header:
            stop("ledger and cut file disagree on columns. The frozen config "
                 "changed, or the ledger was written by a different script; "
                 "reconcile it deliberately rather than appending across it.")
        existing = pd.read_csv(LEDGER, dtype=str)
        already = set(map(tuple, existing[KEY].to_numpy().tolist()))
    else:
        existing = pd.DataFrame(columns=frame.columns)

    keys = [tuple(row) for row in frame[KEY].to_numpy().tolist()]
    fresh = [line for line, key in zip(data_lines, keys) if key not in already]

    print(f"cut file      : {args.cut_file}  ({len(data_lines)} rows)")
    print(f"ledger before : {len(existing)} rows")
    print(f"already held  : {len(data_lines) - len(fresh)} rows (same ticker and "
          f"last_known_period_end)")
    print(f"appending     : {len(fresh)} rows")

    if not fresh:
        # Not a failure. On most days inside earnings season every ticker's
        # last filed quarter is unchanged, so there is nothing new to predict.
        print("nothing new to append; ledger untouched.")
        return

    payload = ""
    if not previous_bytes:
        payload += header + "\n"
    elif not previous_bytes.endswith(b"\n"):
        # Never let an append run onto the end of the last committed row.
        payload += "\n"
    payload += "\n".join(fresh) + "\n"

    with open(LEDGER, "a", newline="") as handle:
        handle.write(payload)

    with open(LEDGER, "rb") as handle:
        written = handle.read()
    if not written.startswith(previous_bytes):
        stop("the ledger's existing bytes changed during the append. Nothing "
             "may rewrite a committed prediction; refusing to leave this in "
             "place silently.")

    print(f"ledger after  : {len(pd.read_csv(LEDGER, dtype=str))} rows -> {LEDGER}")


if __name__ == "__main__":
    main()
