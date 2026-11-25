from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

# =====================================================================
# ORIGINAL CODE (unchanged): load the model and the predict function
# =====================================================================
@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "churn_model.pkl")


classifier = load_model()


def predict_churn(
    Days_Since_Last_Order,
    Orders_Last_90_Days,
    Website_Visits_Last_30_Days,
    App_Sessions_Last_30_Days,
    Total_Spend_AED,
    Complaint_Flag,
    Cart_Abandonment_Rate,
    Support_Tickets_Last_90_Days,
    Email_Open_Rate,
    Membership_Level,
    Subscription_Status,
    Returns_Last_90_Days,
):
    input_data = pd.DataFrame([{
        'Days_Since_Last_Order': Days_Since_Last_Order,
        'Orders_Last_90_Days': Orders_Last_90_Days,
        'Website_Visits_Last_30_Days': Website_Visits_Last_30_Days,
        'App_Sessions_Last_30_Days': App_Sessions_Last_30_Days,
        'Total_Spend_AED': Total_Spend_AED,
        'Complaint_Flag': Complaint_Flag,
        'Cart_Abandonment_Rate': Cart_Abandonment_Rate,
        'Support_Tickets_Last_90_Days': Support_Tickets_Last_90_Days,
        'Email_Open_Rate': Email_Open_Rate,
        'Membership_Level': Membership_Level,
        'Subscription_Status': Subscription_Status,
        'Returns_Last_90_Days': Returns_Last_90_Days,
    }])

    prediction = classifier.predict(input_data)
    return 'Will Churn' if prediction[0] == 1 else 'Will Not Churn'


# =====================================================================
# NEW: settings, example customers and helper functions
# =====================================================================
FEATURES = [
    "Days_Since_Last_Order", "Orders_Last_90_Days", "Website_Visits_Last_30_Days",
    "App_Sessions_Last_30_Days", "Total_Spend_AED", "Complaint_Flag",
    "Cart_Abandonment_Rate", "Support_Tickets_Last_90_Days", "Email_Open_Rate",
    "Membership_Level", "Subscription_Status", "Returns_Last_90_Days",
]

# Starting values are the same as the original app (zeros and first options)
DEFAULTS = {
    "Days_Since_Last_Order": 0, "Orders_Last_90_Days": 0,
    "Website_Visits_Last_30_Days": 0, "App_Sessions_Last_30_Days": 0,
    "Total_Spend_AED": 0.0, "Complaint_Flag": 0, "Cart_Abandonment_Rate": 0.0,
    "Support_Tickets_Last_90_Days": 0, "Email_Open_Rate": 0.0,
    "Membership_Level": "Bronze", "Subscription_Status": "Yes",
    "Returns_Last_90_Days": 0,
}

PRESETS = {
    "Custom (enter your own)": None,
    "Loyal customer": {
        "Days_Since_Last_Order": 5, "Orders_Last_90_Days": 8,
        "Website_Visits_Last_30_Days": 25, "App_Sessions_Last_30_Days": 30,
        "Total_Spend_AED": 3000.0, "Complaint_Flag": 0, "Cart_Abandonment_Rate": 0.10,
        "Support_Tickets_Last_90_Days": 0, "Email_Open_Rate": 0.70,
        "Membership_Level": "Gold", "Subscription_Status": "Yes",
        "Returns_Last_90_Days": 0,
    },
    "At-risk customer": {
        "Days_Since_Last_Order": 120, "Orders_Last_90_Days": 0,
        "Website_Visits_Last_30_Days": 1, "App_Sessions_Last_30_Days": 0,
        "Total_Spend_AED": 50.0, "Complaint_Flag": 1, "Cart_Abandonment_Rate": 0.90,
        "Support_Tickets_Last_90_Days": 4, "Email_Open_Rate": 0.05,
        "Membership_Level": "Bronze", "Subscription_Status": "No",
        "Returns_Last_90_Days": 3,
    },
}

# Test-set results from your notebook (Random Forest). Change them if you switch models.
# Logistic Regression would be: Accuracy 0.81, Precision 0.65, Recall 0.23, F1 0.33
MODEL_METRICS = {
    "Accuracy": 0.79,
    "Precision (churn)": 0.79,
    "Recall (churn)": 0.06,
    "F1-score (churn)": 0.11,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("risk_threshold", 0.30)


def apply_preset():
    preset = PRESETS[st.session_state["preset_choice"]]
    if preset:
        for key, value in preset.items():
            st.session_state[key] = value
        st.session_state["last_result"] = None


def predict_probability(df):
    """Probability (0 to 1) that each customer will churn."""
    return classifier.predict_proba(df[FEATURES])[:, 1]


def risk_level(prob, threshold):
    if prob >= threshold:
        return "High"
    if prob >= threshold / 2:
        return "Medium"
    return "Low"


def make_gauge(prob, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Churn probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#374151"},
            "steps": [
                {"range": [0, threshold * 50], "color": "#bbf7d0"},
                {"range": [threshold * 50, threshold * 100], "color": "#fde68a"},
                {"range": [threshold * 100, 100], "color": "#fecaca"},
            ],
            "threshold": {"line": {"color": "red", "width": 4},
                          "thickness": 0.8, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def get_recommendations(row):
    """Simple business rules that suggest what the shop could do."""
    tips = []
    if row["Complaint_Flag"] == 1:
        tips.append("Customer has complained: contact them personally and fix the issue.")
    if row["Days_Since_Last_Order"] > 60:
        tips.append("No order for over 60 days: send a 'we miss you' discount.")
    if row["Cart_Abandonment_Rate"] > 0.6:
        tips.append("High cart abandonment: send cart reminders or free-shipping offers.")
    if row["Email_Open_Rate"] < 0.2:
        tips.append("Low email engagement: try push notifications or SMS instead.")
    if row["Support_Tickets_Last_90_Days"] >= 3:
        tips.append("Many support tickets: check that their problems were really solved.")
    if row["Returns_Last_90_Days"] >= 3:
        tips.append("Many returns: review product quality or size and fit information.")
    if row["Subscription_Status"] == "No":
        tips.append("Not subscribed: offer a membership or subscription deal.")
    if not tips:
        tips.append("No warning signs found. Keep the customer happy with regular rewards.")
    return tips


def feature_importance(trained_model):
    final = trained_model.steps[-1][1] if hasattr(trained_model, "steps") else trained_model
    try:
        names = trained_model[:-1].get_feature_names_out()
    except Exception:
        return None
    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
    elif hasattr(final, "coef_"):
        values = np.abs(final.coef_[0])
    else:
        return None
    names = [n.split("__", 1)[-1] for n in names]
    df = pd.DataFrame({"Feature": names, "Importance": values})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


# =====================================================================
# TAB 1: PREDICT (the original app, with new extras added underneath)
# =====================================================================
def predict_tab():
    st.write("Enter customer details to predict churn probability.")

    # NEW: quick example customers
    st.selectbox("Quick example (optional)", list(PRESETS.keys()),
                 key="preset_choice", on_change=apply_preset)

    # ---- ORIGINAL inputs (same labels, same order) ----
    Days_Since_Last_Order = st.number_input("Days Since Last Order", min_value=0, step=1,
                                            key="Days_Since_Last_Order")
    Orders_Last_90_Days = st.number_input("Orders in Last 90 Days", min_value=0, step=1,
                                          key="Orders_Last_90_Days")
    Website_Visits_Last_30_Days = st.number_input("Website Visits in Last 30 Days",
                                                  min_value=0, step=1,
                                                  key="Website_Visits_Last_30_Days")
    App_Sessions_Last_30_Days = st.number_input("App Sessions in Last 30 Days",
                                                min_value=0, step=1,
                                                key="App_Sessions_Last_30_Days")
    Total_Spend_AED = st.number_input("Total Spend (AED)", min_value=0.0, format="%.2f",
                                      key="Total_Spend_AED")
    Complaint_Flag = st.selectbox("Complaint Flag", [0, 1], key="Complaint_Flag")
    Cart_Abandonment_Rate = st.slider("Cart Abandonment Rate", 0.0, 1.0, step=0.01,
                                      key="Cart_Abandonment_Rate")
    Support_Tickets_Last_90_Days = st.number_input("Support Tickets in Last 90 Days",
                                                   min_value=0, step=1,
                                                   key="Support_Tickets_Last_90_Days")
    Email_Open_Rate = st.slider("Email Open Rate", 0.0, 1.0, step=0.01,
                                key="Email_Open_Rate")
    Membership_Level = st.selectbox("Membership Level",
                                    ['Bronze', 'Silver', 'Gold', 'Platinum'],
                                    key="Membership_Level")
    Subscription_Status = st.selectbox("Subscription Status", ['Yes', 'No'],
                                       key="Subscription_Status")
    Returns_Last_90_Days = st.number_input("Returns in Last 90 Days", min_value=0, step=1,
                                           key="Returns_Last_90_Days")

    # NEW: setting for the risk level
    with st.expander("Advanced settings"):
        st.slider("High-risk threshold", 0.05, 0.95, step=0.05, key="risk_threshold",
                  help="Customers whose churn probability is at or above this value are "
                       "marked as High risk. Only about 1 in 5 customers churn, so a "
                       "threshold below 0.5 is more useful.")

    # ---- ORIGINAL button and result ----
    if st.button("Predict Churn"):
        result = predict_churn(
            Days_Since_Last_Order,
            Orders_Last_90_Days,
            Website_Visits_Last_30_Days,
            App_Sessions_Last_30_Days,
            Total_Spend_AED,
            Complaint_Flag,
            Cart_Abandonment_Rate,
            Support_Tickets_Last_90_Days,
            Email_Open_Rate,
            Membership_Level,
            Subscription_Status,
            Returns_Last_90_Days,
        )
        customer = {feature: st.session_state[feature] for feature in FEATURES}
        # NEW: also keep the probability so we can show more details
        probability = float(predict_probability(pd.DataFrame([customer]))[0])
        st.session_state["last_result"] = {
            "customer": customer, "label": result, "probability": probability}

    # ---- NEW: extra details shown after a prediction ----
    last = st.session_state["last_result"]
    if last:
        threshold = st.session_state["risk_threshold"]
        prob = last["probability"]
        level = risk_level(prob, threshold)

        st.success(f"Prediction: {last['label']}")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_gauge(prob, threshold), use_container_width=True)
        with right:
            c1, c2 = st.columns(2)
            c1.metric("Churn probability", f"{prob:.1%}")
            c2.metric("Risk level", level)
            st.caption("The prediction above is the model's default decision. The risk "
                       "level uses the high-risk threshold from Advanced settings.")
            st.subheader("Suggested actions")
            for tip in get_recommendations(last["customer"]):
                st.write(f"- {tip}")

        if st.button("💾 Save this prediction to history"):
            record = dict(last["customer"])
            record["Churn_Probability"] = round(prob, 4)
            record["Risk_Level"] = level
            record["Prediction"] = last["label"]
            st.session_state["history"].append(record)
            st.toast("Saved to history")

    if st.session_state["history"]:
        st.subheader("Prediction history")
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(history_df, use_container_width=True)
        h1, h2 = st.columns(2)
        h1.download_button("⬇️ Download history (CSV)", history_df.to_csv(index=False),
                           "prediction_history.csv", "text/csv")
        if h2.button("🗑️ Clear history"):
            st.session_state["history"] = []
            st.rerun()


# =====================================================================
# NEW TAB 2: MODEL INSIGHTS
# =====================================================================
def insights_tab():
    final_step = classifier.steps[-1][1] if hasattr(classifier, "steps") else classifier
    st.subheader("Model")
    st.write(f"Algorithm: **{type(final_step).__name__}** inside a scikit-learn pipeline "
             "(missing values, scaling and one-hot encoding are done automatically).")

    st.subheader("Performance on the test set")
    cols = st.columns(len(MODEL_METRICS))
    for col, (name, value) in zip(cols, MODEL_METRICS.items()):
        col.metric(name, f"{value:.2f}")
    st.info("About 21% of customers in the test data churn, so a model that always says "
            "'will not churn' already gets about 79% accuracy. That is why precision, recall "
            "and F1-score for the churn class matter more than accuracy here.")

    st.subheader("What matters most to the model")
    importance = feature_importance(classifier)
    if importance is None:
        st.write("Feature importance is not available for this model.")
        return
    top = importance.head(15).iloc[::-1]
    fig = px.bar(top, x="Importance", y="Feature", orientation="h", title="Top features")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    best = ", ".join(importance["Feature"].head(3))
    st.write(f"The three most important features are: **{best}**.")


# =====================================================================
# NEW TAB 3: ABOUT
# =====================================================================
def about_tab():
    st.subheader("About this project")
    st.write(
        "This app was built as a machine learning project that follows the CRISP-DM steps: "
        "business understanding, data understanding, data preparation, modeling, "
        "evaluation and deployment.")
    st.markdown("""
**How to use it**
1. Pick a quick example or enter customer details on the *Predict* tab.
2. Click **Predict Churn** to see the result, the probability, the risk level and suggested actions.
3. Check *Model insights* to see how good the model is and which features matter.

**Tools:** Python, pandas, scikit-learn, Plotly, Streamlit, GitHub, Streamlit Community Cloud
""")
    st.caption("The suggested actions are simple business rules, not model output. "
               "Predictions are estimates and should support, not replace, human decisions.")


# =====================================================================
# MAIN
# =====================================================================
def main():
    st.title("Customer Churn Prediction")

    tab_predict, tab_insights, tab_about = st.tabs(
        ["🔮 Predict", "📊 Model insights", "ℹ️ About"])
    with tab_predict:
        predict_tab()
    with tab_insights:
        insights_tab()
    with tab_about:
        about_tab()


if __name__ == '__main__':
    main()
