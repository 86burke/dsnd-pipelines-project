import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load('trained_pipeline.pkl')

# Dashboard title
st.title("Product Recommendation Dashboard")

# Sidebar for user input
st.sidebar.header("Input Review Details")

# Input fields for user data
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
positive_feedback_count = st.sidebar.number_input("Positive Feedback Count", min_value=0, value=0)
division_name = st.sidebar.selectbox("Division Name", ["General", "Petite", "Tall"])
department_name = st.sidebar.selectbox("Department Name", ["Tops", "Bottoms", "Dresses"])
class_name = st.sidebar.selectbox("Class Name", ["Blouses", "Pants", "Skirts"])
review_text = st.sidebar.text_area("Review Text", "Enter the review text here...")

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Age': [age],
    'Positive Feedback Count': [positive_feedback_count],
    'Division Name': [division_name],
    'Department Name': [department_name],
    'Class Name': [class_name],
    'Review Text': [review_text]
})

# Display input data
st.write("### Input Data")
st.write(input_data)

# Predict recommendation
if st.button("Predict Recommendation"):
    prediction = pipeline.predict(input_data)[0]
    prediction_proba = pipeline.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.success(f"The product is recommended with a probability of {prediction_proba:.2f}.")
    else:
        st.error(f"The product is not recommended with a probability of {1 - prediction_proba:.2f}.")