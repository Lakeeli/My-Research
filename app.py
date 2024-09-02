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
image = Image.open('muk.jpeg')
st.sidebar.image(image, width=300)
st.sidebar.title('Upload CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a CSG file", type=["csv"])
st.sidebar.markdown("""
[Example input file](sample.csv)
""")


# Main content
st.write("""

 """)

st.write("""
## HIV LTFU E-SCREENING WEB APPLICATION.

 """)

# Project Description and Column Key
st.write("""
### To use this web application, click BROWSE FILES to upload the CSV file for prediction
### OR 
### DRAG and DROP the CSV file.




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

