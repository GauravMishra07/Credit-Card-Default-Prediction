import streamlit as st
import numpy as np
import pandas as pd 
import pickle

# Function to load the pre-trained model
def load_model():
    with open('pipe.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Main function to run the web app
def main():
    st.title('Credit Card Default Prediction')

    st.write('Enter the values for each column to predict the likelihood of credit card default.')

    # Load the pre-trained model
    model = load_model()

    # Input fields for each column
    inputs = {}
    for column in ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'pay_amt1', 'pay_amt2', 'pay_amt3']:
        inputs[column] = st.number_input(column.capitalize(), step=1)

    # Predict button
    if st.button('Predict'):
        input_values = [inputs[column] for column in inputs]
        input_data = pd.DataFrame([input_values], columns=inputs.keys())
        prediction = model.predict(input_data)[0]
        st.success(f'The predicted likelihood of credit card default is {prediction:.2f}')

if __name__ == '__main__':
    main()
