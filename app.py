import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "xgb_model.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"
DATA_PATH = APP_DIR / "digital_marketing_campaign_dataset1.csv"

FEATURE_COLUMNS = [
    "Age", "Income", "AdSpend", "ClickThroughRate", "ConversionRate",
    "WebsiteVisits", "PagesPerVisit", "TimeOnSite", "SocialShares",
    "EmailOpens", "EmailClicks", "PreviousPurchases", "LoyaltyPoints",
    "Gender_Male", "CampaignChannel_PPC", "CampaignChannel_Referral",
    "CampaignChannel_SEO", "CampaignChannel_Social Media"
]

OPTIMAL_THRESHOLD = 0.15
COST_PER_AD = 100
REVENUE_PER_CONVERSION = 1000

st.set_page_config(
    page_title="Digital Marketing Conversion Prediction",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    return model, scaler

@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

model, scaler = load_model_and_scaler()
df = load_dataset()


@st.cache_data
def get_personas_df():
    if df is None:
        return None
    processed_df = df.copy()
    # Encode Gender
    processed_df['Gender_Male'] = (processed_df['Gender'] == 'Male').astype(int)
    # Encode CampaignChannel
    channels = ['PPC', 'Referral', 'SEO', 'Social Media']
    for channel in channels:
        processed_df[f'CampaignChannel_{channel}'] = (processed_df['CampaignChannel'] == channel).astype(int)
    # Drop original
    processed_df.drop(['Gender', 'CampaignChannel'], axis=1, inplace=True)
    # Select features
    input_features = processed_df[FEATURE_COLUMNS]
    # Scale
    input_scaled = scaler.transform(input_features)
    # Predict
    probs = model.predict_proba(input_scaled)[:, 1]
    # Personas
    personas = [assign_persona(p) for p in probs]
    # Add to original df
    personas_df = df.copy()
    personas_df['Conversion_Probability'] = probs
    personas_df['Persona'] = personas
    return personas_df


def assign_persona(probability: float) -> str:
    if probability >= 0.70:
        return "High Intent"
    if probability >= OPTIMAL_THRESHOLD:
        return "Warm Prospect"
    return "Low Intent"


def recommend_strategy(row: pd.Series, persona: str) -> str:
    if persona == "High Intent":
        if row["PreviousPurchases"] > 3:
            return "Loyalty rewards / premium upsell"
        if row["EmailClicks"] > 5:
            return "High-converting email retargeting"
        if row["TimeOnSite"] > 10:
            return "Personalized product recommendations"
        return "High-budget retargeting campaign"

    if persona == "Warm Prospect":
        if row["EmailClicks"] > 2:
            return "Email nurturing + discount offer"
        if row["TimeOnSite"] > 7:
            return "Remarketing ads"
        return "Awareness + engagement campaign"

    return "Do not target / low-cost awareness only"


def build_input_dataframe(values: dict) -> pd.DataFrame:
    campaign = values.pop("CampaignChannel")
    gender = values.pop("Gender")

    row = {
        **values,
        "Gender_Male": 1 if gender == "Male" else 0,
        "CampaignChannel_PPC": 1 if campaign == "PPC" else 0,
        "CampaignChannel_Referral": 1 if campaign == "Referral" else 0,
        "CampaignChannel_SEO": 1 if campaign == "SEO" else 0,
        "CampaignChannel_Social Media": 1 if campaign == "Social Media" else 0,
    }

    input_df = pd.DataFrame([row])
    return input_df[FEATURE_COLUMNS]


def predict_conversion(input_df: pd.DataFrame):
    input_scaled = scaler.transform(input_df)
    probability = float(model.predict_proba(input_scaled)[0, 1])
    decision = "TARGET" if probability >= OPTIMAL_THRESHOLD else "DO NOT TARGET"
    persona = assign_persona(probability)
    strategy = recommend_strategy(input_df.iloc[0], persona)
    expected_revenue = probability * REVENUE_PER_CONVERSION
    expected_profit = expected_revenue - COST_PER_AD
    expected_roas = expected_revenue / COST_PER_AD
    return probability, decision, persona, strategy, expected_revenue, expected_profit, expected_roas


st.title("📈 Digital Marketing Campaign Conversion Prediction")
st.caption("XGBoost-powered targeting, customer intent scoring, and budget allocation intelligence")

with st.sidebar:
    st.header("Customer Input")
    st.write("Enter customer and campaign details below.")

    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 69, 35)
    income = st.number_input("Income", min_value=0, max_value=200000, value=85000, step=1000)

    campaign_channel = st.selectbox(
        "Campaign Channel",
        ["Email", "PPC", "Referral", "SEO", "Social Media"]
    )

    ad_spend = st.number_input("Ad Spend", min_value=0.0, max_value=20000.0, value=5000.0, step=100.0)
    click_through_rate = st.slider("Click Through Rate", 0.00, 0.50, 0.15, 0.01)
    conversion_rate = st.slider("Historical Conversion Rate", 0.00, 0.30, 0.10, 0.01)

    website_visits = st.slider("Website Visits", 0, 100, 25)
    pages_per_visit = st.slider("Pages Per Visit", 0.0, 15.0, 5.5, 0.1)
    time_on_site = st.slider("Time On Site", 0.0, 20.0, 7.5, 0.1)

    social_shares = st.slider("Social Shares", 0, 150, 50)
    email_opens = st.slider("Email Opens", 0, 30, 9)
    email_clicks = st.slider("Email Clicks", 0, 20, 4)
    previous_purchases = st.slider("Previous Purchases", 0, 20, 4)
    loyalty_points = st.number_input("Loyalty Points", min_value=0, max_value=10000, value=2500, step=100)

values = {
    "Age": age,
    "Gender": gender,
    "Income": income,
    "CampaignChannel": campaign_channel,
    "AdSpend": ad_spend,
    "ClickThroughRate": click_through_rate,
    "ConversionRate": conversion_rate,
    "WebsiteVisits": website_visits,
    "PagesPerVisit": pages_per_visit,
    "TimeOnSite": time_on_site,
    "SocialShares": social_shares,
    "EmailOpens": email_opens,
    "EmailClicks": email_clicks,
    "PreviousPurchases": previous_purchases,
    "LoyaltyPoints": loyalty_points,
}

input_df = build_input_dataframe(values.copy())
probability, decision, persona, strategy, expected_revenue, expected_profit, expected_roas = predict_conversion(input_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversion Probability", f"{probability:.2%}")
col2.metric("Decision", decision)
col3.metric("Persona", persona)
col4.metric("Expected ROAS", f"{expected_roas:.2f}x")

if decision == "TARGET":
    st.success(f"Recommended Action: {strategy}")
else:
    st.warning(f"Recommended Action: {strategy}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prediction Details", "Business Impact", "Feature Importance", "Dataset Overview", "User Personas"
])

with tab1:
    st.subheader("Customer Decision Output")
    output = input_df.copy()
    output.insert(0, "conversion_prob", probability)
    output.insert(1, "persona", persona)
    output.insert(2, "decision", decision)
    output.insert(3, "recommended_strategy", strategy)
    st.dataframe(output, use_container_width=True)

    st.markdown(
        f"""
        **How to read this:**  
        The model estimates this customer has a **{probability:.2%}** chance of converting.  
        Since the business threshold is **{OPTIMAL_THRESHOLD:.2f}**, the system recommends: **{decision}**.
        """
    )

with tab2:
    st.subheader("Expected Financial Value")
    b1, b2, b3 = st.columns(3)
    b1.metric("Expected Revenue", f"₹{expected_revenue:,.0f}")
    b2.metric("Estimated Campaign Cost", f"₹{COST_PER_AD:,.0f}")
    b3.metric("Expected Profit", f"₹{expected_profit:,.0f}")

    st.markdown(
        """
        This section converts the probability score into a business view.  
        The app assumes **₹100 cost per targeted customer** and **₹1000 revenue per conversion**.
        """
    )

    chart_df = pd.DataFrame({
        "Metric": ["Expected Revenue", "Campaign Cost", "Expected Profit"],
        "Value": [expected_revenue, COST_PER_AD, expected_profit]
    })
    st.bar_chart(chart_df.set_index("Metric"))

with tab3:
    st.subheader("XGBoost Feature Importance")
    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance["Feature"][::-1], importance["Importance"][::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Feature Importance - XGBoost")
    st.pyplot(fig)

    st.markdown(
        """
        Feature importance shows which variables the model relied on most when predicting conversion.  
        In marketing, these features help explain the strongest conversion drivers.
        """
    )

with tab4:
    st.subheader("Dataset Overview")
    if df is not None:
        d1, d2, d3 = st.columns(3)
        d1.metric("Rows", f"{df.shape[0]:,}")
        d2.metric("Columns", f"{df.shape[1]:,}")
        d3.metric("Conversion Rate", f"{df['Conversion'].mean():.2%}")

        st.write("Sample data")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("Conversion Distribution")
        conv_counts = df["Conversion"].value_counts().sort_index()
        conv_counts.index = ["Non-Converter", "Converter"]
        st.bar_chart(conv_counts)
    else:
        st.info("Dataset file not found. The prediction app still works with the saved model and scaler.")

with tab5:
    st.subheader("User Personas Overview")
    personas_df = get_personas_df()
    if personas_df is not None:
        high_intent = personas_df[personas_df['Persona'] == 'High Intent']
        warm_prospect = personas_df[personas_df['Persona'] == 'Warm Prospect']
        low_intent = personas_df[personas_df['Persona'] == 'Low Intent']
        st.write("### High Intent Users")
        st.dataframe(high_intent.head(50), use_container_width=True)
        st.write("### Warm Prospect Users")
        st.dataframe(warm_prospect.head(50), use_container_width=True)
        st.write("### Low Intent Users")
        st.dataframe(low_intent.head(50), use_container_width=True)
    else:
        st.info("Dataset not available.")

st.divider()
st.caption("Capstone Project: Digital Marketing Campaign Conversion Prediction | Final Model: XGBoost")
