"""Unit tests for data loader module."""

from pathlib import Path
import pandas as pd
import pytest

from workforce_scheduler.data_loader import (
    load_tabular_data,
    inspect_excel_sheets,
    compute_file_hash
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_load_sample_workers_csv():
    csv_path = DATA_DIR / "sample_workers.csv"
    assert csv_path.exists(), "sample_workers.csv should exist"

    df, meta = load_tabular_data(csv_path, csv_path.name)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert "worker_id" in df.columns
    assert "preferred_weekly_hours" in df.columns
    assert "sha256" in meta
    assert meta["extension"] == ".csv"


def test_load_sample_availability_csv():
    csv_path = DATA_DIR / "sample_availability.csv"
    assert csv_path.exists()

    df, meta = load_tabular_data(csv_path, csv_path.name)
    assert len(df) == 1020  # 20 workers * 51 slots
    assert "worker_id" in df.columns
    assert "available" in df.columns


def test_inspect_and_load_excel_sheets():
    xlsx_path = DATA_DIR / "sample_workforce.xlsx"
    assert xlsx_path.exists()

    sheets = inspect_excel_sheets(xlsx_path)
    assert "Workers" in sheets
    assert "Availability" in sheets
    assert "Demand" in sheets

    df_workers, meta = load_tabular_data(xlsx_path, xlsx_path.name, sheet_name="Workers")
    assert len(df_workers) == 20
    assert meta["selected_sheet"] == "Workers"


def test_unsupported_file_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_tabular_data(b"test content", "invalid_file.xyz")
