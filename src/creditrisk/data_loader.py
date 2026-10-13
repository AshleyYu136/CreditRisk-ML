"""Download and validate the original UCI spreadsheet; no synthetic fallback."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from creditrisk.config import DATA

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
EXPECTED_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
TARGET = "default"
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_STATUS, *BILLS, *PAYMENTS]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE", *PAY_STATUS]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(destination: Path | None = None) -> Path:
    destination = destination or DATA / "raw/default of credit card clients.xls"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = destination.with_suffix(".source.json")
    if destination.exists():
        if sha256(destination) != EXPECTED_SHA256:
            raise ValueError(
                "Dataset is not the expected original UCI spreadsheet (SHA-256 mismatch)"
            )
        if manifest.exists() and json.loads(manifest.read_text())["sha256"] != sha256(destination):
            raise ValueError("Cached dataset checksum mismatch. Remove it and download again.")
        validate(load_data(destination), expected_rows=30000)
        return destination
    archive = destination.parent / "uci.zip"
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        archive.write_bytes(content)
        with zipfile.ZipFile(archive) as bundle:
            names = [n for n in bundle.namelist() if n.lower().endswith(".xls")]
            if len(names) != 1:
                raise ValueError("Expected exactly one UCI XLS dataset")
            destination.write_bytes(bundle.read(names[0]))
        if sha256(destination) != EXPECTED_SHA256:
            raise ValueError("Downloaded dataset SHA-256 does not match the recorded UCI source")
        validate(load_data(destination), expected_rows=30000)
        manifest.write_text(json.dumps({"url": URL, "sha256": sha256(destination)}, indent=2))
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "UCI download failed. Download the original XLS from "
            "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients "
            f"and place it at {destination}. Original error: {exc}"
        ) from exc
    finally:
        archive.unlink(missing_ok=True)
    return destination


def load_data(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA / "raw/default of credit card clients.xls"
    frame = pd.read_excel(path, header=1).rename(columns={"default payment next month": TARGET})
    validate(frame)
    return frame


def validate(frame: pd.DataFrame, expected_rows: int | None = None) -> dict:
    required = ["ID", *FEATURES, TARGET]
    if set(frame.columns) != set(required):
        raise ValueError(f"Dataset columns differ: {set(frame.columns) ^ set(required)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; got {len(frame)}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Dataset must be nonempty and contain no missing values")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in required):
        raise ValueError("All original columns must be numeric")
    if not np.isfinite(frame[required].to_numpy()).all():
        raise ValueError("Nonfinite data")
    if not set(frame[TARGET].unique()) <= {0, 1} or frame[TARGET].nunique() != 2:
        raise ValueError("Target must contain both binary classes")
    allowed = {"SEX": {1, 2}, "EDUCATION": set(range(7)), "MARRIAGE": set(range(4))}
    allowed.update({c: set(range(-2, 10)) for c in PAY_STATUS})
    for col, codes in allowed.items():
        if not set(frame[col].unique()) <= codes:
            raise ValueError(f"Unexpected categorical codes: {col}")
    if frame.ID.duplicated().any():
        raise ValueError("Duplicate customer IDs")
    if (frame.AGE < 18).any() or (frame.AGE > 100).any() or (frame.LIMIT_BAL <= 0).any():
        raise ValueError("Invalid age or credit limit")
    if (frame[PAYMENTS] < 0).any().any():
        raise ValueError("Negative payment amounts")
    numeric = frame[["LIMIT_BAL", "AGE", *BILLS, *PAYMENTS]]
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    outliers = ((numeric < q1 - 1.5 * (q3 - q1)) | (numeric > q3 + 1.5 * (q3 - q1))).sum()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "default_rate": float(frame[TARGET].mean()),
        "missing": int(frame.isna().sum().sum()),
        "duplicate_records_excluding_id": int(frame.drop(columns="ID").duplicated().sum()),
        "ambiguous_education_codes": int(frame.EDUCATION.isin([0, 5, 6]).sum()),
        "ambiguous_marriage_codes": int((frame.MARRIAGE == 0).sum()),
        "iqr_outliers": {k: int(v) for k, v in outliers.items()},
        "dtypes": frame.dtypes.astype(str).to_dict(),
    }
