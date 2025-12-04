import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime


st.set_page_config(
    page_title="AutoVision AI",
    page_icon="🚗",
    layout="wide",
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("car.csv")

    df["car_name"] = (
        df["Brand"].astype(str)
        + " "
        + df["Model"].astype(str)
        + " (" + df["Year"].astype(str) + ")"
    )

    df.rename(columns={
        "Brand": "brand",
        "Year": "year",
        "Mileage": "km_driven",        
        "Fuel Type": "fuel_type",
        "Transmission": "transmission",
        "Price": "selling_price",
    }, inplace=True)

    # car age
    current_year = datetime.now().year
    df["car_age"] = current_year - df["year"]

    df["fuel_efficiency"] = (df["km_driven"].max() - df["km_driven"]) / 5000
    df["fuel_efficiency"] = df["fuel_efficiency"].clip(lower=5, upper=25)

    #future price
    df["future_price"] = df["selling_price"] * 0.8

    return df


df = load_data()

# SIDE NAVbar
st.sidebar.title("AutoVision AI")
section = st.sidebar.radio(
    "Go to:",
    [
        "🏁 Intro & Goals",
        "🔮 Prediction Demo",
        "📊 Charts & Analytics",
        "🤝 Car Recommendation System",
    ],
)

# 1) INTRO & GOALS
if section == "🏁 Intro & Goals":
    st.title("AutoVision AI 🚗")
    st.subheader("A Smart Car Analytics & Prediction Platform")

    st.markdown(
        """
        **AutoVision AI** is a demo web platform built on a car dataset.
        It aims to:
        - Explore and visualize car data  
        - Show example prediction outputs (price, age, efficiency, future value)  
        - Provide a simple rule-based car recommendation experience  
        """
    )

    st.markdown("### Dataset Snapshot")
    st.dataframe(df.head())

# 2) PREDICTION DEMO
elif section == "🔮 Prediction Demo":
    st.title("Prediction Demo 🔮")
    st.write(
        "Choose a car from the dataset(Demo)"
    )

    car_options = df["car_name"].tolist()
    selected_car = st.selectbox("Choose a car from the dataset:", car_options)

    car_row = df[df["car_name"] == selected_car].iloc[0]

    selling_price_example = float(car_row["selling_price"])
    car_age_example = float(car_row["car_age"])
    fuel_efficiency_example = float(car_row["fuel_efficiency"])
    future_price_example = float(car_row["future_price"])

    st.markdown("### Demo Cards")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.write("**Selling Price (Demo)**")
        st.button(
            f"{selling_price_example:,.0f} (USD)",
            help="Example predicted selling price.",
        )

    with c2:
        st.write("**Car Age (Demo)**")
        st.button(
            f"{car_age_example:.1f} years",
            help="Example predicted car age.",
        )

    with c3:
        st.write("**Fuel Efficiency (Demo)**")
        st.button(
            f"{fuel_efficiency_example:.1f} km/L",
            help="Example predicted fuel efficiency.",
        )

    with c4:
        st.write("**Future Price (Demo)**")
        st.button(
            f"{future_price_example:,.0f} (USD)",
            help="Example predicted future resale value.",
        )

    st.info("Right now these are static demo values based on the row data")

# 3) CHARTS & ANALYTICS
elif section == "📊 Charts & Analytics":
    st.title("Charts & Analytics 📊")
    st.write("Explore the dataset using filters and interactive charts.")

    # Filters
    brands = ["All"] + sorted(df["brand"].unique().tolist())
    fuels = ["All"] + sorted(df["fuel_type"].unique().tolist())
    transmissions = ["All"] + sorted(df["transmission"].unique().tolist())

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_brand = st.selectbox("Filter by brand:", brands)
    with col_f2:
        selected_fuel = st.selectbox("Filter by fuel type:", fuels)
    with col_f3:
        selected_trans = st.selectbox("Filter by transmission:", transmissions)

    filtered_df = df.copy()
    if selected_brand != "All":
        filtered_df = filtered_df[filtered_df["brand"] == selected_brand]
    if selected_fuel != "All":
        filtered_df = filtered_df[filtered_df["fuel_type"] == selected_fuel]
    if selected_trans != "All":
        filtered_df = filtered_df[filtered_df["transmission"] == selected_trans]

    # Chart 1: Average Selling Price by Brand
    st.markdown("### 1) Average Selling Price by Brand (Bar Chart)")

    avg_price = (
        filtered_df.groupby("brand")["selling_price"]
        .mean()
        .reset_index()
        .sort_values("selling_price", ascending=False)
    )

    bar_chart = (
        alt.Chart(avg_price)
        .mark_bar()
        .encode(
            x=alt.X("brand:N", sort="-y", title="Brand"),
            y=alt.Y("selling_price:Q", title="Average Selling Price"),
            tooltip=["brand", "selling_price"],
        )
        .properties(height=400)
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # Chart 2: Km Driven vs Selling Price
    st.markdown("### 2) Km Driven vs Selling Price (Scatter Plot)")

    scatter = (
        alt.Chart(filtered_df)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X("km_driven:Q", title="Km Driven"),
            y=alt.Y("selling_price:Q", title="Selling Price"),
            color=alt.Color("fuel_type:N", title="Fuel Type"),
            tooltip=["car_name", "km_driven", "selling_price"],
        )
        .properties(height=400)
    )
    st.altair_chart(scatter, use_container_width=True)

# 4) CAR RECOMMENDATION SYSTEM
elif section == "🤝 Car Recommendation System":
    st.title("Car Recommendation System 🤝")
    st.write(
        "Answer a few questions and AutoVision AI will suggest some cars that match your preferences. "
        "This is a simple rule-based demo — not a full AI recommender yet."
    )

    with st.form("recommendation_form"):
        budget = st.number_input(
            "Budget (same unit as Price column):",
            min_value=1000.0,
            max_value=float(df["selling_price"].max() * 2),
            value=float(df["selling_price"].median()),
            step=1000.0,
        )

        usage_type = st.selectbox(
            "Usage type:",
            ["City", "Travel", "Family", "Mixed"],
        )

        fuel_pref = st.selectbox(
            "Fuel preference:",
            ["Any"] + sorted(df["fuel_type"].unique().tolist()),
        )

        trans_pref = st.selectbox(
            "Transmission preference:",
            ["Any"] + sorted(df["transmission"].unique().tolist()),
        )

        year_min, year_max = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider(
            "Preferred year range:",
            year_min,
            year_max,
            (max(year_min, year_max - 10), year_max),
        )

        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        rec_df = df.copy()

        rec_df = rec_df[(rec_df["year"] >= year_range[0]) & (rec_df["year"] <= year_range[1])]
        rec_df = rec_df[rec_df["selling_price"] <= budget * 1.2]
        if fuel_pref != "Any":
            rec_df = rec_df[rec_df["fuel_type"] == fuel_pref]
        if trans_pref != "Any":
            rec_df = rec_df[rec_df["transmission"] == trans_pref]

        if rec_df.empty:
            st.warning("No cars found that match your filters. Try adjusting budget or year range.")
        else:
            rec_df["budget_diff"] = (rec_df["selling_price"] - budget).abs()
            rec_df["score"] = -rec_df["budget_diff"] + rec_df["year"] * 10

            top_cars = rec_df.sort_values("score", ascending=False).head(3)

            st.subheader("Best-matching cars:")
            for _, row in top_cars.iterrows():
                st.markdown("---")
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"### 🚗 {row['car_name']}")
                    st.markdown(f"**Estimated Price:** {row['selling_price']:,.0f}")
                    st.markdown(f"**Year:** {int(row['year'])}")
                    st.markdown(f"**Km Driven:** {int(row['km_driven']):,}")
                    st.markdown(f"**Fuel Type:** {row['fuel_type']}")
                    st.markdown(f"**Transmission:** {row['transmission']}")

                with col2:
                    reasons = []
                    if row["selling_price"] <= budget:
                        reasons.append("Fits within your budget.")
                    else:
                        reasons.append("Slightly above your budget but still close.")
                    if usage_type == "City":
                        reasons.append("Suitable for daily city driving.")
                    elif usage_type == "Travel":
                        reasons.append("Good option for long-distance travel.")
                    elif usage_type == "Family":
                        reasons.append("Suitable for family usage.")
                    else:
                        reasons.append("Balanced choice for mixed usage.")
                    if fuel_pref != "Any":
                        reasons.append("Matches your fuel preference.")

                    st.markdown("**Why this car?**")
                    for r in reasons:
                        st.markdown(f"- {r}")

            st.success("These recommendations are rule-based for demo purposes")
