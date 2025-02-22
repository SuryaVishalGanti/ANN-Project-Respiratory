import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import sqlite3

# Set up SQLite connection
def create_connection():
    conn = sqlite3.connect('customer_data.db')  # This creates the database file
    return conn

# Function to create the table (if not already created)
def create_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS customer_data
                 (name TEXT, geography TEXT, gender TEXT, age INTEGER, balance REAL, 
                  credit_score INTEGER, estimated_salary REAL, tenure INTEGER, 
                  num_of_products INTEGER, has_cr_card INTEGER, is_active_member INTEGER)''')
    conn.commit()
    conn.close()

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gen.pkl', 'rb') as file:
    label_encoder_gen = pickle.load(file)

with open('one_hot_en_geo.pkl', 'rb') as file:
    one_hot_en_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Customer churn prediction')

# Input Fields
name = st.text_input("Name of the User:")
geography = st.selectbox('Geography', one_hot_en_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gen.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance', min_value=0.0, step=100.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900)
estimated_salary = st.number_input('Estimated Salary per month', min_value=0.0, step=1000.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, value=5)
num_of_products = st.slider('Number of Products', 1, 5, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Save button
save_button = st.button("Save")

# Create table if it doesn't exist
create_table()

# Preparing the input data
try:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gen.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = one_hot_en_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_en_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.write(f'Churn Probability: {prediction_proba: .2f}')

    if prediction_proba > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

    # If save button is pressed, save the input data to the database
    if save_button:
        conn = create_connection()
        c = conn.cursor()
        c.execute('''INSERT INTO customer_data (name, geography, gender, age, balance, credit_score, estimated_salary, 
                  tenure, num_of_products, has_cr_card, is_active_member) 
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (name, geography, gender, age, balance, credit_score, estimated_salary, tenure, 
                   num_of_products, has_cr_card, is_active_member))
        conn.commit()
        conn.close()
        st.success("Data saved successfully!")

except Exception as e:
    st.error(f"An error occurred: {e}")
