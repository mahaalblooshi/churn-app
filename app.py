
import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

classifier = joblib.load('churn_model.pkl')

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
    Returns_Last_90_Days
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
        'Returns_Last_90_Days': Returns_Last_90_Days
    }])

    prediction = classifier.predict(input_data)
    label = 'Will Churn' if prediction[0] == 1 else 'Will Not Churn'
    return label

def main():
    st.title("Customer Churn Prediction")
    html_temp = """
        <div style="background-color:red">
            <h2 style="color:white;text-align:center;">This our Streamlit Deployment App</h2>
        </div>
        """
    st.write("Enter customer details to predict churn probability.")

    Days_Since_Last_Order = st.number_input("Days Since Last Order", min_value=0)
    Orders_Last_90_Days = st.number_input("Orders in Last 90 Days", min_value=0)
    Website_Visits_Last_30_Days = st.number_input("Website Visits in Last 30 Days", min_value=0)
    App_Sessions_Last_30_Days = st.number_input("App Sessions in Last 30 Days", min_value=0)
    Total_Spend_AED = st.number_input("Total Spend (AED)", min_value=0.0, format="%.2f")
    Complaint_Flag = st.selectbox("Complaint Flag", [0, 1])
    Cart_Abandonment_Rate = st.slider("Cart Abandonment Rate", 0.0, 1.0, step=0.01)
    Support_Tickets_Last_90_Days = st.number_input("Support Tickets in Last 90 Days", min_value=0)
    Email_Open_Rate = st.slider("Email Open Rate", 0.0, 1.0, step=0.01)
    Membership_Level = st.selectbox("Membership Level", ['Bronze', 'Silver', 'Gold', 'Platinum'])
    Subscription_Status = st.selectbox("Subscription Status", ['Yes', 'No'])
    Returns_Last_90_Days = st.number_input("Returns in Last 90 Days", min_value=0)

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
            Returns_Last_90_Days
        )
        st.success(f"Prediction: {result}")

if __name__ == '__main__':
    main()
    
