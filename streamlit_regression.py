import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd

scaler = StandardScaler()


# Load the trained model
try:
    model = tf.keras.models.load_model('regression.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load the encoders and scaler
try:
    with open('label_encoder_gen.pkl', 'rb') as file:
        label_encoder_gen = pickle.load(file)

    with open('one_hot_en_geo.pkl', 'rb') as file:
        one_hot_en_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the encoders or scaler: {e}")

# Streamlit App
st.title('Customer Churn Prediction')

# Input Fields
name = st.text_input("Name of the User:")
geography = st.selectbox('Geography', one_hot_en_geo.categories_[0] if 'one_hot_en_geo' in locals() else [])
gender = st.selectbox('Gender', label_encoder_gen.classes_ if 'label_encoder_gen' in locals() else [])
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance', min_value=0.0, step=100.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900)
estimated_salary = st.number_input('Estimated Salary per month', min_value=0.0, step=1000.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, value=5)
num_of_products = st.slider('Number of Products', 1, 5, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Preparing the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})
# Save button (optional, for saving input data in the app, not to a database)
save_button = st.button("Save")
# One-hot encode 'Geography'
geo_encoded = one_hot_en_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_en_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Check if the scaler is fitted before transforming
if hasattr(scaler, 'scale_'):
    input_data_scaled = scaler.transform(input_data)
else:
    st.error("Scaler is not fitted. Please check the scaler or retrain the model.")
    input_data_scaled = None

if input_data_scaled is not None:
    # Predict the churn (regression model)
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Churn Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')
