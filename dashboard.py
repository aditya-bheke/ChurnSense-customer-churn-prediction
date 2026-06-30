import streamlit as st
import pandas as pd
import requests
import os
from PIL import Image

st.set_page_config(page_title="ChurnSense Dashboard", layout="wide")

st.title("📞 ChurnSense | Customer Churn Prediction Dashboard")

st.markdown("""
This dashboard visualizes the customer churn dataset, model performance metrics, and allows you to predict whether a customer will churn in real-time.
""")

tab1, tab2, tab3 = st.tabs(["Dashboard & Metrics", "Data Exploration", "Real-time Prediction"])

REPORT_DIR = "reports"

with tab1:
    st.header("Model Performance & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), use_column_width=True)
        else:
            st.info("Run the training pipeline to generate the confusion matrix plot.")

    with col2:
        st.subheader("Feature Importance")
        fi_path = os.path.join(REPORT_DIR, "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(Image.open(fi_path), use_column_width=True)
        else:
            st.info("Run the training pipeline to generate the feature importance plot.")

with tab2:
    st.header("Data Exploration")
    
    data_path = "data/churn_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        st.bar_chart(churn_counts)
        
    else:
        st.info("Run the training pipeline to generate the dataset.")

with tab3:
    st.header("Real-time Prediction")
    st.markdown("Enter customer details to get a real-time prediction from the FastAPI service.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=70.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)
            num_services = st.number_input("Number of Add-on Services", min_value=1, max_value=8, value=3)
        
        with col2:
            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            contract = st.selectbox("Contract Type", list(contract_map.keys()))
            
            internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
            internet = st.selectbox("Internet Service", list(internet_map.keys()))
            
            payment_map = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}
            payment = st.selectbox("Payment Method", list(payment_map.keys()))
            
        with col3:
            tech_map = {"No": 0, "Yes": 1, "No internet service": 2}
            tech_support = st.selectbox("Tech Support", list(tech_map.keys()))
            
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            senior_val = 1 if senior == "Yes" else 0
            
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            dependents_val = 1 if dependents == "Yes" else 0
            
        submit = st.form_submit_button("Predict Churn")
        
    if submit:
        payload = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract_map[contract],
            "InternetService": internet_map[internet],
            "PaymentMethod": payment_map[payment],
            "TechSupport": tech_map[tech_support],
            "SeniorCitizen": senior_val,
            "Dependents": dependents_val,
            "NumServices": num_services
        }
        
        try:
            # Assuming FastAPI is running locally on port 8000
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                
                st.subheader("Prediction Result")
                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", result["churn_prediction"])
                col2.metric("Probability", f"{result['churn_probability']*100:.1f}%")
                col3.metric("Risk Level", result["risk_level"])
                
                if result["churn_prediction"] == "CHURN":
                    st.error("This customer is at HIGH risk of churning.")
                else:
                    st.success("This customer is likely to be retained.")
            else:
                st.error(f"API Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the Prediction API. Please ensure FastAPI is running (uvicorn app:app --reload) on port 8000.")

