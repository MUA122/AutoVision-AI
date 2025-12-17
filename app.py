# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from recommender import train_recommender, recommend

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="AutoVision AI | Car Recommendation",
    page_icon="🚗",
    layout="wide",
)

DATA_PATH = "data.csv"


# ---------------------------
# PDF Export
# ---------------------------
def build_recommendations_pdf(user_prefs: dict, recs_df: pd.DataFrame, reasons: list) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "AutoVision AI - Car Recommendations")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 25

    # User Prefs
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "User Preferences")
    y -= 18

    c.setFont("Helvetica", 11)
    for k, v in user_prefs.items():
        c.drawString(60, y, f"- {k}: {v}")
        y -= 15

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Top Recommendations")
    y -= 20

    # Recommendations
    for i, row in recs_df.reset_index(drop=True).iterrows():
        if y < 140:
            c.showPage()
            y = height - 50

        car_name = row.get("car_name", "Car")
        price = row.get("Price", None)
        year = row.get("Year", None)
        fuel = row.get("Fuel Type", "")
        trans = row.get("Transmission", "")
        sim = row.get("similarity", None)

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{i+1}) {car_name}")
        y -= 16

        c.setFont("Helvetica", 11)
        line1 = " | ".join(
            [
                f"Price: {float(price):,.0f}" if price is not None and str(price) != "nan" else "Price: -",
                f"Year: {int(year)}" if year is not None and str(year) != "nan" else "Year: -",
                f"Fuel: {fuel}" if fuel else "Fuel: -",
                f"Transmission: {trans}" if trans else "Transmission: -",
            ]
        )
        c.drawString(60, y, line1)
        y -= 14

        if sim is not None and str(sim) != "nan":
            c.drawString(60, y, f"Similarity score: {float(sim):.3f}")
            y -= 14

        if i < len(reasons):
            c.setFont("Helvetica-Bold", 11)
            c.drawString(60, y, "Reasons:")
            y -= 14
            c.setFont("Helvetica", 11)
            for r in reasons[i]:
                if y < 80:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 11)
                c.drawString(75, y, f"- {r}")
                y -= 13

        y -= 10

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------
# Prepare Artifacts
# ---------------------------
@st.cache_resource
def ensure_artifacts(csv_path: str):
    train_recommender(csv_path, force=False)
    return True


# ---------------------------
# UI
# ---------------------------
st.title("🚗 AutoVision AI — Car Recommendation System")
st.caption("ML-based content recommendation using vector similarity (encoding + scaling + cosine similarity).")

if not Path(DATA_PATH).exists():
    st.error(f"Couldn't find `{DATA_PATH}` beside app.py. Please add it to the project folder.")
    st.stop()

df = load_data(DATA_PATH)

required_cols = {"Price", "Year", "Brand", "Fuel Type", "Transmission"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Dataset is missing required columns: {sorted(list(missing))}")
    st.stop()

try:
    ensure_artifacts(DATA_PATH)
except Exception as e:
    st.error("Failed to prepare recommendation artifacts.")
    st.exception(e)
    st.stop()

# Sidebar quiz
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

usage_type = st.sidebar.selectbox("Usage type", ["City", "Travel", "Family", "Mixed"])

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

# Main content
colA, colB = st.columns([2, 1])

with colB:
    with st.expander("Dataset quick info", expanded=False):
        st.write(f"Rows: **{len(df):,}**")
        st.write(f"Columns: **{len(df.columns):,}**")

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

    # Keep results so user can export without re-running (session state)
    st.session_state["top_cars"] = top_cars.copy()
    st.session_state["reasons"] = reasons

# Display if we have results
if "top_cars" in st.session_state and not st.session_state["top_cars"].empty:
    top_cars = st.session_state["top_cars"].reset_index(drop=True)
    reasons = st.session_state.get("reasons", [])

    with colA:
        st.subheader("✅ Recommended Cars")

        for i, row in top_cars.iterrows():
            st.markdown("---")
            left, right = st.columns([2, 1])

            with left:
                car_title = row.get("car_name", f"{row.get('Brand','')} {row.get('Model','')}")
                st.markdown(f"### 🚘 {car_title}")

                # Core fields
                st.write(f"**Estimated Price:** {float(row['Price']):,.0f}")
                st.write(f"**Year:** {int(row['Year'])}")
                st.write(f"**Fuel Type:** {row.get('Fuel Type', '-')}")
                st.write(f"**Transmission:** {row.get('Transmission', '-')}")

                # Optional fields if present
                for optional_col in ["Mileage", "Engine Size", "Condition"]:
                    if optional_col in row.index and pd.notna(row[optional_col]):
                        st.write(f"**{optional_col}:** {row[optional_col]}")

                if "similarity" in row and pd.notna(row["similarity"]):
                    st.caption(f"Similarity score: {float(row['similarity']):.3f}")

            with right:
                st.markdown("**Why this car?**")
                if i < len(reasons):
                    for r in reasons[i]:
                        st.write(f"- {r}")
                else:
                    st.write("- Matches your preferences closely.")

                # Show extra engineered signals if present (optional)
                extra_cols = [
                    "value_score", "wear_score", "maintenance_index",
                    "yearly_fuel_cost", "is_suv", "is_luxury"
                ]
                extras = [(c, row[c]) for c in extra_cols if c in row.index and pd.notna(row[c])]
                if extras:
                    st.markdown("**Extra signals:**")
                    for k, v in extras[:6]:
                        st.write(f"- {k}: {v}")

    # Export PDF
    user_prefs = {
        "Budget": f"{float(budget):,.0f}",
        "Usage type": usage_type,
        "Fuel preference": fuel_pref,
        "Transmission preference": trans_pref,
        "Year range": f"{year_range[0]} - {year_range[1]}",
        "Top K": str(top_k),
    }

    pdf_bytes = build_recommendations_pdf(user_prefs, top_cars, reasons)

    st.download_button(
        label="📄 Export Recommendations as PDF",
        data=pdf_bytes,
        file_name="autovision_recommendations.pdf",
        mime="application/pdf",
    )

else:
    st.info("Set your preferences from the sidebar, then click **Recommend Cars**.")
