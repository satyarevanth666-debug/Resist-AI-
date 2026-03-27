from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Target mapping
TARGET_MAPPING = {
    "Resistant": 1,
    "Susceptible": 0,
    "Intermediate": 2
}


def load_and_prepare_data(
    primary_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:

    # File path
    primary_path = primary_path or (DATA_DIR / "primary.xlsx")

    if not primary_path.exists():
        raise FileNotFoundError(f"Missing file: {primary_path}")

    # Load data
    primary_df = pd.read_excel(primary_path)

    print("📂 Original Columns:", primary_df.columns)

    # 🔥 STEP 1: Convert wide → long format
    if "Location" in primary_df.columns:
        primary_df = primary_df.melt(
            id_vars=["Location"],
            var_name="Antibiotic",
            value_name="Result"
        )
        primary_df = primary_df.rename(columns={"Location": "Bacteria"})

    # Ensure correct columns
    primary_df = primary_df[["Bacteria", "Antibiotic", "Result"]]

    # ============================================================
    # 🔥 STEP 2: HANDLE MIC NUMERIC VALUES (MAIN FIX)
    # ============================================================

    # Convert to numeric safely
    primary_df["Result"] = pd.to_numeric(primary_df["Result"], errors="coerce")

    # Drop NaN values
    primary_df = primary_df.dropna()

    # Convert MIC → Classes
    def mic_to_label(value):
        if value <= 20:
            return "Susceptible"
        elif value <= 30:
            return "Intermediate"
        else:
            return "Resistant"

    primary_df["Result"] = primary_df["Result"].apply(mic_to_label)

    print("✅ Converted labels:", primary_df["Result"].unique())

    # ============================================================
    # 🔥 STEP 3: CLEAN DATA
    # ============================================================

    primary_df = primary_df.drop_duplicates()

    primary_df = primary_df[
        primary_df["Result"].isin(TARGET_MAPPING.keys())
    ].copy()

    # 🚨 Safety check
    if primary_df.empty:
        raise ValueError(
            "❌ No valid rows after cleaning.\n"
            "👉 Check your Excel values."
        )

    # ============================================================
    # 🔥 STEP 4: CHECK CLASS DISTRIBUTION
    # ============================================================

    print("\n📊 Class distribution:")
    print(primary_df["Result"].value_counts())

    if primary_df["Result"].nunique() < 2:
        raise ValueError(
            "❌ ERROR: Only ONE class present after cleaning.\n"
            "👉 Your dataset must contain at least 2 classes."
        )

    # ============================================================
    # 🔥 STEP 5: ENCODE TARGET
    # ============================================================

    primary_df["ResultCode"] = primary_df["Result"].map(TARGET_MAPPING)

    # ============================================================
    # 🔥 STEP 6: ENCODE FEATURES
    # ============================================================

    bacteria_encoder = LabelEncoder()
    antibiotic_encoder = LabelEncoder()

    primary_df["BacteriaEncoded"] = bacteria_encoder.fit_transform(primary_df["Bacteria"])
    primary_df["AntibioticEncoded"] = antibiotic_encoder.fit_transform(primary_df["Antibiotic"])

    encoders = {
        "bacteria_encoder": bacteria_encoder,
        "antibiotic_encoder": antibiotic_encoder,
    }

    # Features (X)
    X = primary_df[["BacteriaEncoded", "AntibioticEncoded"]]

    return primary_df, X, encoders