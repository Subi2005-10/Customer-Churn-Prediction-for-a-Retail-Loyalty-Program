"""Load the three raw tables from the FreshBasket workbook.

Each data sheet has a title row above the real header, so the header is on row 2.
"""
import pandas as pd

from .config import RAW_FILE, SHEETS


def load_raw(path=RAW_FILE) -> dict:
    tables = {}
    for key, sheet in SHEETS.items():
        df = pd.read_excel(path, sheet_name=sheet, header=1)
        df.columns = [str(c).strip().upper() for c in df.columns]
        tables[key] = df
    # Type coercion only (no cleaning here)
    tables["profile"]["SIGNUP_DATE"] = pd.to_datetime(tables["profile"]["SIGNUP_DATE"])
    tables["activity"]["MONTH"] = pd.to_datetime(tables["activity"]["MONTH"])
    for c in ["OBSERVATION_END_DATE", "LAST_PURCHASE_DATE"]:
        tables["label"][c] = pd.to_datetime(tables["label"][c])
    return tables
