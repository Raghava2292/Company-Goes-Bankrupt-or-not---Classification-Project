# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:19:29 2023

@author: Raghava Varanasi
"""
## Import the libraries
import pandas as pd
import streamlit as st 
from pickle import load
import joblib
from keras.models import load_model
import numpy as np

# Title of the website
st.title('Model Deployment: Company goes Bankrupt or Not')

# Choosing between predicting a single observation or an entire dataset
st.subheader('Single Prediction or Dataset Prediction')
predict_option = st.radio('Select One Option:', ('Single Prediction', 'Dataset Prediction'))

# Choosing a regressor
st.subheader('Select the Classification Model')
classifier = st.selectbox('Available Models arranged in descending order of their prediction accuracies: ', ('AdaBoost', 'Decision Tree', 'Random Forest', 'Bagging', 'KNN', 'Stacking', 'Gradient Boost', 'Support Vector - RBF Kernel', 'Support Vector - Polynomial Kernel', 'Neural Networks', 'Logistic Regression', 'Support Vector - Linear Kernel'))

# Loading the model based on the user selection
if classifier == 'AdaBoost':
    loaded_model = load(open('AdaBoost.sav', 'rb'))
    st.subheader('AdaBoost Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Decision Tree':
    loaded_model = load(open('Decision_Tree.sav', 'rb'))
    st.subheader('Decision Tree Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Random Forest':
    loaded_model = load(open('Random_Forest.sav', 'rb'))
    st.subheader('Random Forest Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Bagging':
    loaded_model = load(open('Bagging.sav', 'rb'))
    st.subheader('Bagging Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Gradient Boost':
    loaded_model = load(open('Gradient_Boost.sav', 'rb'))
    st.subheader('Gradient Boost Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Stacking':
    loaded_model = load(open('Stacking.sav', 'rb'))
    st.subheader('Stacking Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'KNN':
    loaded_model = load(open('KNN.sav', 'rb'))
    st.subheader('KNN Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Logistic Regression':
    loaded_model = load(open('Logistic_Regression_model.sav', 'rb'))
    st.subheader('Logistic Regression Classification Model')
    st.markdown('Model Score   --   **99.651%**')
elif classifier == 'Neural Networks':
    loaded_model = load_model('ANN.h5')
    st.subheader('Neural Network Classification Model')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Support Vector - RBF Kernel':
    loaded_model = load(open('SVC_RBF.sav', 'rb'))
    st.subheader('Support Vector Classification Model - RBF Kernel')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Support Vector - Polynomial Kernel':
    loaded_model = load(open('SVC_Poly.sav', 'rb'))
    st.subheader('Support Vector Classification Model - Polynomial Kernel')
    st.markdown('Model Score   --   **100%**')
elif classifier == 'Support Vector - Linear Kernel':
    loaded_model = load(open('SVC_Linear.sav', 'rb'))
    st.subheader('Support Vector Classification Model - Linear Kernel')
    st.markdown('Model Score   --   **99.651%**')

# Single prediction
if predict_option == 'Single Prediction':
    st.sidebar.header('User Input Parameters - Select values between 0 to 1:')
    
    ind_risk = st.sidebar.slider('Industrial Risk:', 0.0, 1.0, step = 0.5)
    man_risk = st.sidebar.slider('Management Risk:', 0.0, 1.0, step = 0.5)
    fin_flex = st.sidebar.slider('Financial Flexibility:', 0.0, 1.0, step = 0.5)
    cred = st.sidebar.slider('Credibilty:', 0.0, 1.0, step = 0.5)
    comp = st.sidebar.slider('Competitiveness:', 0.0, 1.0, step = 0.5)
    op_risk = st.sidebar.slider('Operating Risk:', 0.0, 1.0, step = 0.5)
    
    data = {'Industrial Risk':ind_risk,
            'Management Risk':man_risk,
            'Financial Flexibility':fin_flex,
            'Credibility':cred,
            'Competitiveness':comp,
            'Operating Risk':op_risk}
    df = pd.DataFrame(data,index = [0])
        
    st.subheader('User Input parameters')
    st.write(df)
    st.subheader('Make Prediction:')
    if st.button('Predict'):
        st.subheader(f'Predicted Result - {classifier} Classifier')
        result = loaded_model.predict(df)[0]
        if result == 0:
            st.markdown('The company **:red[will go bankrupt]**')
        elif result == 1:
            st.markdown('The company **:green[will not go bankrupt]**')

# Dataset prediction
else:
    st.subheader('Upload the dataset')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None: 
        df = pd.read_csv(uploaded_file)
        df = df[['Industrial Risk', 'Management Risk', 'Financial Flexibility', 'Credibility', 'Competitiveness', 'Operating Risk']]
        
        st.subheader('Make Predictions:')
        if st.button('Predict'):
            prediction = loaded_model.predict(df)
            st.subheader(f'Predicted Result - {classifier} Classifier')

            df['Class Code'] = prediction
            
            df['Class'] = np.where(df['Class Code'] == 0, 'Bankrupt', 'Not Bankrupt')
            st.write(df)
            
            st.subheader('Download Predictions:')
            st.download_button('Download Predictions', data = df.to_csv().encode('utf-8'), file_name=f'Precited Data - {classifier} Classifier.csv', mime='text/csv')


        












    