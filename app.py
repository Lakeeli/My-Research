# pip install streamlit pycaret

import streamlit as st
import pandas as pd
import joblib
from io import StringIO
from PIL import Image

# Load the saved model
model = joblib.load('lgmb_model.pkl')

# Set up the sidebar
# Logo image
image = Image.open('muk.jpg')
st.sidebar.image(image, width=300)
st.sidebar.title('Upload CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
st.sidebar.markdown("""
[Example input file](sample.csv)
""")


# Main content
st.write("""
## MAKERERE UNIVERSITY SCHOOL OF PUBLIC HEALTH
 """)

st.write("""
### A PREDICTION MODEL OF NEWLY DIAGNOSED HIV PATIENTS LIKELY TO FALL OFF TREATMENT.  A CASE OF MUKONO GENERAL REFERRAL HOSPITAL.

 """)

# Project Description and Column Key
st.write("""
#### Background
Introduction
In spite of all the measures put in place to reduce the problem of lost to follow-up among PLHIV in Uganda for example the use of simple messaging service (SMS) to remind the patients about their next appointment among others, these have not effectively solved the problem. However, retaining PLWH in medical care is paramount to preventing new transmissions of the virus and allowing PLWH to live normal and healthy lifespans. While much effort and resources have been focused on tracing those LTFU and returning them to care, very little prior work has successfully addressed identifying those most at risk of dropping out of care while still engaged in care. Thus, this study aimed to develop a web-based application to predict newly diagnosed HIV patients likely to fall off treatment while still in care using machine learning.

General Objective
To develop a predictive model for classifying patients to serve as a tool for health workers in predicting the newly diagnosed patients likely to fall off treatment at Mukono General Referral Hospital.

Specific Objective
1.	To identify factors that lead to lost to follow-up during HIV treatment.
2.	To build HIV retention in care predictive machine learning models and evaluate their performance.
3.	To develop a web-based interface for utilization of the identified best prediction model.
""")


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Display the uploaded data
    st.write('Uploaded Data:')
    st.write(data)

    # Make predictions using the loaded model and add a 'Prediction' column
    predictions = model.predict(data)  # Assuming 'model' is a trained model object
    data['Prediction'] = predictions

    # Display the predictions
    st.write('Predictions:')
    st.write(data)

    # Export results to a CSV file
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=StringIO(csv).read(),
        file_name='predictions.csv',
        mime='text/csv'
    )

