# recommender.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity


ARTIFACTS_DIR = "artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
CAR_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "car_matrix.joblib")
CARS_DF_PATH = os.path.join(ARTIFACTS_DIR, "cars_df.joblib")

ID_COLS = {"Car ID", "car_id", "id"}  # columns to exclude from vectors


def _ensure_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Create a nice car_name
    if {"Brand", "Model", "Year"}.issubset(df.columns):
        df["car_name"] = df["Brand"].astype(str) + " " + df["Model"].astype(str) + " (" + df["Year"].astype(str) + ")"
    elif "Brand" in df.columns and "Model" in df.columns:
        df["car_name"] = df["Brand"].astype(str) + " " + df["Model"].astype(str)
    else:
        df["car_name"] = df.index.astype(str)

    return df


def train_recommender(csv_path: str, force: bool = False):
    """
    Builds:
      - preprocessor: encodes categorical + scales numerical
      - car_matrix: vector representation for each car
      - cars_df: cleaned dataframe
    """
    _ensure_dir()

    if (not force) and all(os.path.exists(p) for p in [PREPROCESSOR_PATH, CAR_MATRIX_PATH, CARS_DF_PATH]):
        return

    df = pd.read_csv(csv_path)
    df = _standardize(df)

    # Required columns for filtering + UI
    required = ["Price", "Year", "Fuel Type", "Transmission", "Brand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in your data.csv: {missing}")

    # Convert common numeric columns safely
    for col in df.columns:
        if col in ["Year", "Price", "Mileage", "Engine Size", "engine_cc", "mileage_val"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essential columns
    df = df.dropna(subset=["Price", "Year", "Brand", "Fuel Type", "Transmission"]).reset_index(drop=True)

    # --------- Auto-detect feature columns ----------
    # We'll exclude ID columns and also exclude the raw Price from vectors (use it only for filtering & display)
    excluded = set([c for c in df.columns if c in ID_COLS]) | {"Price"}

    feature_cols = [c for c in df.columns if c not in excluded and c != "car_name"]

    # Categorical vs Numerical detection
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if df[c].dtype != "object"]

    # --------- Preprocessing Pipelines ----------
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )

    # Fit and transform -> vectors
    car_matrix = preprocessor.fit_transform(df[feature_cols])

    # Save artifacts
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(car_matrix, CAR_MATRIX_PATH)
    joblib.dump(df, CARS_DF_PATH)


def _load_artifacts():
    if not all(os.path.exists(p) for p in [PREPROCESSOR_PATH, CAR_MATRIX_PATH, CARS_DF_PATH]):
        raise FileNotFoundError("Artifacts not found. Run train_recommender('data.csv') once.")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    car_matrix = joblib.load(CAR_MATRIX_PATH)
    cars_df = joblib.load(CARS_DF_PATH)
    return preprocessor, car_matrix, cars_df


def recommend(
    budget: float,
    usage_type: str,
    fuel_pref: str,
    trans_pref: str,
    year_min: int,
    year_max: int,
    top_k: int = 5
):
    """
    ML content-based recommendation:
      1) filter by budget/year/fuel/trans
      2) build a "user vector" from preferences (same feature space)
      3) cosine similarity vs filtered cars
      4) return top_k with reasons
    """
    preprocessor, car_matrix, df = _load_artifacts()

    # --- Hard filters (guarantee different answers -> different candidates) ---
    candidates = df.copy()
    candidates = candidates[(candidates["Year"] >= year_min) & (candidates["Year"] <= year_max)]
    candidates = candidates[candidates["Price"] <= budget * 1.2]  # +20% tolerance

    if fuel_pref != "Any":
        candidates = candidates[candidates["Fuel Type"].astype(str).str.lower() == fuel_pref.lower()]

    if trans_pref != "Any":
        candidates = candidates[candidates["Transmission"].astype(str).str.lower() == trans_pref.lower()]

    if candidates.empty:
        return candidates, []

    # --- Determine the exact feature columns used during training ---
    excluded = set([c for c in df.columns if c in ID_COLS]) | {"Price"}
    feature_cols = [c for c in df.columns if c not in excluded and c != "car_name"]

    # --- Build a user row in the SAME columns ---
    # We set only what user chose; rest become NaN -> handled by imputers.
    user_row = {c: np.nan for c in feature_cols}

    # Map user answers to dataset columns (if present in features)
    if "Fuel Type" in feature_cols and fuel_pref != "Any":
        user_row["Fuel Type"] = fuel_pref
    if "Transmission" in feature_cols and trans_pref != "Any":
        user_row["Transmission"] = trans_pref

    # Usage type can influence some engineered flags if they exist:
    # City -> prefer not SUV (is_suv=0), Family/Travel -> maybe SUV=1 (as a soft hint)
    if "is_suv" in feature_cols:
        if usage_type == "City":
            user_row["is_suv"] = 0
        elif usage_type in ["Family", "Travel"]:
            user_row["is_suv"] = 1

    # Year preference: choose mid of range as a numeric hint if year exists as engineered feature (not raw "Year")
    # Raw "Year" is excluded from features? No, we keep Year because it's not excluded. But we excluded only Price & ID.
    if "Year" in feature_cols:
        user_row["Year"] = (year_min + year_max) / 2

    # Mileage hint (optional): city often lower mileage preference
    if "Mileage" in feature_cols:
        if usage_type == "City":
            user_row["Mileage"] = candidates["Mileage"].median() * 0.8 if "Mileage" in candidates.columns else np.nan
        else:
            user_row["Mileage"] = candidates["Mileage"].median() if "Mileage" in candidates.columns else np.nan

    user_df = pd.DataFrame([user_row])

    # Transform user -> vector
    user_vec = preprocessor.transform(user_df)

    # Similarity only on candidate rows
    cand_idx = candidates.index.to_numpy()
    cand_matrix = car_matrix[cand_idx]

    sims = cosine_similarity(user_vec, cand_matrix).ravel()
    top_local = np.argsort(-sims)[:top_k]
    top_rows = candidates.iloc[top_local].copy()
    top_rows["similarity"] = sims[top_local]

    # Reasons (simple explainability)
    reasons = []
    for _, r in top_rows.iterrows():
        why = []
        if r["Price"] <= budget:
            why.append("Fits your budget")
        else:
            why.append("Close to your budget")
        why.append(f"Year: {int(r['Year'])}")
        why.append(f"Fuel: {r['Fuel Type']}")
        why.append(f"Transmission: {r['Transmission']}")
        reasons.append(why)

    return top_rows, reasons
