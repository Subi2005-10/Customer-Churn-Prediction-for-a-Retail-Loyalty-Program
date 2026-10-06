"""Central configuration: paths, dates and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

SHEETS = {
    "profile": "fb Customer Profile",
    "activity": "fb Monthly Activity",
    "label": "fb Churn Label",
}

# Churn definition: zero transactions in the HORIZON months after the cutoff,
# despite purchase history on or before the cutoff.
HORIZON_MONTHS = 3

# Official label: features use data up to 2024-03 (inclusive); label window Apr-Jun 2024.
TEST_CUTOFF = "2024-03-01"
# Earlier snapshots rebuilt from activity data with the same churn definition.
TRAIN_CUTOFFS = ["2023-06-01", "2023-09-01"]
VALID_CUTOFF = "2023-12-01"

# Columns in fb Churn Label that are computed over the FULL window (incl. Apr-Jun 2024)
# and therefore leak the target. Never used as features.
LEAKY_LABEL_COLUMNS = ["LAST_PURCHASE_DATE", "MONTHS_OBSERVED", "INSUFFICIENT_HISTORY_FLAG"]

SEED = 42

for d in (PROCESSED_DIR, OUTPUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
