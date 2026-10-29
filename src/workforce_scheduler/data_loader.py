"""
Data loader module for reading CSV and Excel files safely.
Supports multiple sheets and file provenance hashing.
"""

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


def compute_file_hash(file_bytes_or_path: Union[bytes, Path, str]) -> str:
    """Compute SHA-256 checksum for provenance tracking."""
    hasher = hashlib.sha256()
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        hasher.update(file_bytes_or_path)
    else:
        path = Path(file_bytes_or_path)
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def inspect_excel_sheets(file_source: Union[bytes, Path, str]) -> List[str]:
    """Return sheet names of an Excel workbook."""
    if isinstance(file_source, (bytes, bytearray)):
        excel_file = pd.ExcelFile(io.BytesIO(file_source))
    else:
        excel_file = pd.ExcelFile(Path(file_source))
    return excel_file.sheet_names


def load_tabular_data(
    file_source: Union[bytes, Path, str],
    file_name: str,
    sheet_name: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load tabular data from CSV or Excel.
    Returns (DataFrame, metadata_dict).
    """
    ext = Path(file_name).suffix.lower()
    meta = {
        "file_name": file_name,
        "extension": ext,
        "sheet_name": sheet_name,
    }

    if isinstance(file_source, (bytes, bytearray)):
        meta["sha256"] = compute_file_hash(file_source)
        meta["size_bytes"] = len(file_source)
        source = io.BytesIO(file_source)
    else:
        path = Path(file_source)
        meta["sha256"] = compute_file_hash(path)
        meta["size_bytes"] = path.stat().st_size
        source = path

    if ext in [".xlsx", ".xls", ".xlsm"]:
        excel = pd.ExcelFile(source)
        target_sheet = sheet_name or excel.sheet_names[0]
        df = pd.read_excel(excel, sheet_name=target_sheet)
        meta["all_sheets"] = excel.sheet_names
        meta["selected_sheet"] = target_sheet
    elif ext == ".csv" or ext == ".txt":
        df = pd.read_csv(source)
        meta["all_sheets"] = [file_name]
        meta["selected_sheet"] = file_name
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Accepted formats are .csv, .xlsx, .xls.")

    meta["num_rows"] = len(df)
    meta["num_cols"] = len(df.columns)
    meta["columns"] = list(df.columns)

    return df, meta
