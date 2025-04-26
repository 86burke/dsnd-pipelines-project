import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load("trained_pipeline.pkl")

# Load the dataset for exploration
data = pd.read_csv("starter/data/reviews.csv")

# Dashboard Title
st.title("Customer Review Recommendation Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Data Exploration", "Make Predictions"])

# Data Exploration Section
if options == "Data Exploration":
    st.header("Explore the Dataset")
    
    # Display the dataset
    st.write("### Dataset Preview")
    st.dataframe(data.head())
    
    # Display basic statistics
    st.write("### Dataset Statistics")
    st.write(data.describe())
    
    # Visualize the distribution of the target variable
    st.write("### Target Variable Distribution")
    st.bar_chart(data["Recommended IND"].value_counts())

# Prediction Section
elif options == "Make Predictions":
    st.header("Make Predictions")
    
    # Input fields for user data
    st.write("Enter the details of the review below:")
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    division_name = st.selectbox("Division Name", data["Division Name"].unique())
    department_name = st.selectbox("Department Name", data["Department Name"].unique())
    class_name = st.selectbox("Class Name", data["Class Name"].unique())
    review_text = st.text_area("Review Text", "Enter the review text here...")
    positive_feedback_count = st.number_input("Positive Feedback Count", min_value=0, value=0)
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        "Age": [age],
        "Division Name": [division_name],
        "Department Name": [department_name],
        "Class Name": [class_name],
        "Review Text": [review_text],
        "Positive Feedback Count": [positive_feedback_count]
    })
    
    # Predict button
    if st.button("Predict"):
        prediction = pipeline.predict(input_data)
        if prediction[0] == 1:
            st.success("The customer is likely to recommend this product!")
        else:
            st.error("The customer is unlikely to recommend this product.")