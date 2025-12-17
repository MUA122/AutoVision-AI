import streamlit as st
import pandas as pd
from pathlib import Path

from recommender import train_recommender, recommend

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="AutoVision AI | Car Recommendation",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 AutoVision AI — Car Recommendation System")
st.caption("ML-based content recommendation using vector similarity (categorical encoding + numeric scaling + cosine similarity).")

DATA_PATH = "data.csv"


# ---------------------------
# Load dataset for UI options
# ---------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


if not Path(DATA_PATH).exists():
    st.error(f"Couldn't find `{DATA_PATH}` in your project folder. Please add it beside app.py.")
    st.stop()

df = load_data(DATA_PATH)

# Basic checks
required_cols = {"Price", "Year", "Brand", "Fuel Type", "Transmission"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Your dataset is missing required columns: {sorted(list(missing))}")
    st.stop()


# ---------------------------
# Train / Load recommender artifacts
# ---------------------------
@st.cache_resource
def ensure_artifacts():
    # Creates artifacts if missing (safe to call on deploy)
    train_recommender(DATA_PATH, force=False)
    return True


try:
    ensure_artifacts()
except Exception as e:
    st.error("Failed to prepare recommendation artifacts.")
    st.exception(e)
    st.stop()


# ---------------------------
# Sidebar: User Preferences (Quiz)
# ---------------------------
st.sidebar.header("🧩 Your Preferences")

price_min = float(max(0, df["Price"].min()))
price_max = float(df["Price"].max())
price_med = float(df["Price"].median())

budget = st.sidebar.number_input(
    "Budget",
    min_value=price_min,
    max_value=float(price_max * 2),
    value=price_med,
    step=max(100.0, price_max * 0.01),
)

usage_type = st.sidebar.selectbox(
    "Usage type",
    ["City", "Travel", "Family", "Mixed"],
)

fuel_options = ["Any"] + sorted(df["Fuel Type"].dropna().astype(str).unique().tolist())
fuel_pref = st.sidebar.selectbox("Fuel preference", fuel_options)

trans_options = ["Any"] + sorted(df["Transmission"].dropna().astype(str).unique().tolist())
trans_pref = st.sidebar.selectbox("Transmission preference", trans_options)

year_min = int(df["Year"].min())
year_max = int(df["Year"].max())

year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(max(year_min, year_max - 10), year_max),
)

top_k = st.sidebar.slider("Number of recommendations", min_value=3, max_value=10, value=5)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔎 Recommend Cars")


# ---------------------------
# Main: Show Recommendations
# ---------------------------
colA, colB = st.columns([2, 1])

with colA:
    st.subheader("✅ Recommended Cars")

with colB:
    with st.expander("Dataset quick info", expanded=False):
        st.write(f"Rows: **{len(df):,}**")
        st.write(f"Columns: **{len(df.columns):,}**")
        st.write("Sample columns:", ", ".join(list(df.columns[:10])) + (" ..." if len(df.columns) > 10 else ""))

if run_btn:
    try:
        top_cars, reasons = recommend(
            budget=float(budget),
            usage_type=str(usage_type),
            fuel_pref=str(fuel_pref),
            trans_pref=str(trans_pref),
            year_min=int(year_range[0]),
            year_max=int(year_range[1]),
            top_k=int(top_k),
        )
    except Exception as e:
        st.error("Recommendation failed.")
        st.exception(e)
        st.stop()

    if top_cars is None or top_cars.empty:
        st.warning("No cars matched your filters. Try increasing budget or widening year range.")
        st.stop()

    # Render results
    top_cars = top_cars.reset_index(drop=True)

    for i, row in top_cars.iterrows():
        st.markdown("---")
        left, right = st.columns([2, 1])

        with left:
            # car_name is generated inside recommender training
            car_title = row.get("car_name", f"{row.get('Brand','')} {row.get('Model','')}")
            st.markdown(f"### 🚘 {car_title}")

            st.write(f"**Estimated Price:** {float(row['Price']):,.0f}")
            st.write(f"**Year:** {int(row['Year'])}")
            st.write(f"**Fuel Type:** {row.get('Fuel Type', '-')}")
            st.write(f"**Transmission:** {row.get('Transmission', '-')}")

            # Optional fields if exist
            if "Mileage" in row and pd.notna(row["Mileage"]):
                try:
                    st.write(f"**Mileage:** {float(row['Mileage']):,.0f}")
                except:
                    st.write(f"**Mileage:** {row['Mileage']}")

            if "Engine Size" in row and pd.notna(row["Engine Size"]):
                st.write(f"**Engine Size:** {row['Engine Size']}")

            # Similarity score (added by recommender)
            if "similarity" in row:
                st.caption(f"Similarity score: {float(row['similarity']):.3f}")

        with right:
            st.markdown("**Why this car?**")
            if i < len(reasons):
                for r in reasons[i]:
                    st.write(f"- {r}")
            else:
                st.write("- Matches your preferences closely.")

            # Extra explainability if engineered columns exist
            extra_signals = []
            for col in ["value_score", "wear_score", "maintenance_index", "yearly_fuel_cost", "is_suv", "is_luxury"]:
                if col in row.index and pd.notna(row[col]):
                    extra_signals.append((col, row[col]))

            if extra_signals:
                st.markdown("**Extra signals (from dataset):**")
                for k, v in extra_signals[:6]:
                    st.write(f"- {k}: {v}")

else:
    st.info("Set your preferences from the sidebar, then click **Recommend Cars**.")
