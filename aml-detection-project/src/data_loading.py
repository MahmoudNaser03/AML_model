"""
Load and normalise HI-Small transaction CSV.
"""

import gc
import re
import csv

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import TRANS_CSV, PATTERNS_TXT


# ── Column dtypes expected in the raw CSV ─────────────────────────────────────
_DTYPES = {
    "From Bank":          "string",
    "Account":            "string",
    "To Bank":            "string",
    "Account.1":          "string",
    "Amount Received":    "float64",
    "Receiving Currency": "string",
    "Amount Paid":        "float64",
    "Payment Currency":   "string",
    "Payment Format":     "string",
    "Is Laundering":      "int8",
}

_RENAME = {
    "Timestamp":          "ts",
    "From Bank":          "from_bank",
    "Account":            "from_account",
    "To Bank":            "to_bank",
    "Account.1":          "to_account",
    "Amount Received":    "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid":        "amount_paid",
    "Payment Currency":   "payment_currency",
    "Payment Format":     "payment_format",
    "Is Laundering":      "target",
}

_REQUIRED = list(_RENAME.values())


def load_transactions(path=None) -> pd.DataFrame:
    """Read, rename, type-coerce and sort the HI-Small transactions CSV."""
    path = path or TRANS_CSV

    parts = []
    for chunk in tqdm(
        pd.read_csv(path, dtype=_DTYPES, chunksize=500_000, low_memory=False),
        desc="Reading HI-Small CSV", unit="chunk",
    ):
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    df = df.rename(columns=_RENAME)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["ts"] = pd.to_datetime(df["ts"], errors="raise")
    for c in ["from_bank", "from_account", "to_bank", "to_account",
              "receiving_currency", "payment_currency", "payment_format"]:
        df[c] = df[c].astype("string").str.strip()

    df["amount_received"] = pd.to_numeric(df["amount_received"], errors="raise")
    df["amount_paid"]     = pd.to_numeric(df["amount_paid"],     errors="raise")
    df["target"]          = pd.to_numeric(df["target"],          errors="raise").astype("int8")

    df["tx_id"]  = np.arange(len(df), dtype=np.int64)
    df           = df.sort_values(["ts", "tx_id"], kind="mergesort").reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df


# ── Pattern-file parser ───────────────────────────────────────────────────────

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT\s*-\s*(.+?)\s*$", re.I)
_END   = re.compile(r"^END LAUNDERING ATTEMPT\s*-\s*(.+?)\s*$",   re.I)


def parse_pattern_file(path=None) -> list[dict]:
    """Return a list of laundering-attempt dicts with keys pattern_id, pattern_type, rows."""
    path = path or PATTERNS_TXT

    attempts, current_type, current_rows = [], None, []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in tqdm(fh, desc="Parsing pattern file", unit="line"):
            line = raw.strip()
            if not line:
                continue
            m = _BEGIN.match(line)
            if m:
                if current_type is not None:
                    raise ValueError("Nested BEGIN pattern block.")
                current_type, current_rows = m.group(1).strip().upper(), []
                continue
            m = _END.match(line)
            if m:
                if current_type is None:
                    raise ValueError("END without BEGIN.")
                attempts.append({"pattern_id": len(attempts),
                                  "pattern_type": current_type,
                                  "rows": current_rows})
                current_type = None
                continue
            if current_type is not None:
                current_rows.append(line)

    if current_type is not None:
        raise ValueError("Pattern file ended before END.")

    return attempts


# ── Pattern-to-row mapper ─────────────────────────────────────────────────────

def _norm_text(x):
    return "" if pd.isna(x) else str(x).strip()


def _norm_num(x):
    try:
        return f"{float(str(x).strip()):.6f}"
    except Exception:
        return _norm_text(x)


def _make_key(ts, fb, fa, tb, ta, ar, rc, ap, pc, pf):
    return (
        pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
        _norm_text(fb), _norm_text(fa),
        _norm_text(tb), _norm_text(ta),
        _norm_num(ar),  _norm_text(rc),
        _norm_num(ap),  _norm_text(pc), _norm_text(pf),
    )


def build_pattern_map(df: pd.DataFrame, attempts: list[dict]) -> pd.DataFrame:
    """Map each laundering-attempt row to transaction row_ids.

    Returns a DataFrame with columns: row_id, pattern_id, pattern_type.
    """
    sig_to_rows: dict[tuple, list[int]] = {}
    for start in tqdm(range(0, len(df), 500_000),
                      desc="Indexing transaction signatures", unit="chunk"):
        chunk = df.iloc[start:start + 500_000]
        for rid, row in chunk.iterrows():
            key = _make_key(
                row.ts, row.from_bank, row.from_account,
                row.to_bank, row.to_account,
                row.amount_received, row.receiving_currency,
                row.amount_paid, row.payment_currency, row.payment_format,
            )
            sig_to_rows.setdefault(key, []).append(int(row.row_id))

    records, unmatched, ambiguous = [], 0, 0
    for attempt in tqdm(attempts, desc="Mapping laundering patterns", unit="attempt"):
        for line in attempt["rows"]:
            try:
                fields = next(csv.reader([line]))
                if len(fields) < 10:
                    unmatched += 1
                    continue
                key = _make_key(*fields[:10])
                matches = sig_to_rows.get(key, [])
                if len(matches) > 1:
                    ambiguous += 1
                for rid in matches:
                    records.append({"row_id": rid,
                                    "pattern_id": attempt["pattern_id"],
                                    "pattern_type": attempt["pattern_type"]})
                if not matches:
                    unmatched += 1
            except Exception:
                unmatched += 1

    if not records:
        raise RuntimeError("Pattern mapping produced zero rows.")

    pattern_map = pd.DataFrame(records).drop_duplicates()

    label_check = df.set_index("row_id").loc[pattern_map.row_id.unique(), "target"]
    if label_check.mean() < 0.99:
        raise ValueError("Pattern mapping contains non-laundering rows.")

    print(f"Pattern records : {len(pattern_map):,}")
    print(f"Pattern tx      : {pattern_map.row_id.nunique():,}")
    print(f"Pattern attempts: {pattern_map.pattern_id.nunique():,}")
    print(f"Unmatched rows  : {unmatched}")
    print(f"Ambiguous       : {ambiguous}")
    return pattern_map
