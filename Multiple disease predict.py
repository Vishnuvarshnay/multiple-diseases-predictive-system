# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:12:12 2023

@author: Admin
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


#Loading the saved models


diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("heart_disease_model.sav", 'rb'))
parkinsons_model = pickle.load(open("parkinsons_model.sav", 'rb'))


#Sidebar for navigation




with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons = ['activity','heart','person'],
                           default_index = 0)
    
    
#Diabetes Prediction Page
if(selected == 'Diabetes Prediction'):
    
    # Page title
 st.markdown("<h2 style='text-align:center; color:#FF4B4B;'>🩺 Diabetes Prediction using Machine Learning</h2>", unsafe_allow_html=True)

# Input columns
 col1, col2 = st.columns(2)

 with col1:
    Pregnancies = st.text_input('Number of Pregnancies')
    BloodPressure = st.text_input('Blood Pressure value')
    Insulin = st.text_input('Insulin Level')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

 with col2:
    Glucose = st.text_input('Glucose Level')
    SkinThickness = st.text_input('Skin Thickness value')
    BMI = st.text_input('BMI value')
    Age = st.text_input('Age of the Person')

# Prediction and confidence
 diab_diagnosis = ''

 if st.button('Diabetes Test Result'):
    try:
        input_data = [float(Pregnancies), float(Glucose), float(BloodPressure),
                      float(SkinThickness), float(Insulin), float(BMI),
                      float(DiabetesPedigreeFunction), float(Age)]

        with st.spinner('Predicting...'):
            diab_prediction = diabetes_model.predict([input_data])
            confidence = diabetes_model.predict_proba([input_data])[0][1]

        if diab_prediction[0] == 1:
            diab_diagnosis = f'The person is Diabetic. Confidence: {confidence * 100:.2f}%'
            st.error(diab_diagnosis)
        else:
            diab_diagnosis = f'The person is Not Diabetic. Confidence: {(1 - confidence) * 100:.2f}%'
            st.success(diab_diagnosis)

    except ValueError:
        st.warning("⚠️ Please enter valid numeric values for all input fields.")

# Bar chart for input summary
 try:
    if st.button("Show Input Summary"):
        input_dict = {
            'Pregnancies': float(Pregnancies),
            'Glucose': float(Glucose),
            'BloodPressure': float(BloodPressure),
            'SkinThickness': float(SkinThickness),
            'Insulin': float(Insulin),
            'BMI': float(BMI),
            'DiabetesPedigreeFunction': float(DiabetesPedigreeFunction),
            'Age': float(Age)
        }
        input_df = pd.DataFrame(input_dict.items(), columns=["Feature", "Value"])
        st.bar_chart(input_df.set_index("Feature"))
 except ValueError:
    st.warning("Please enter all values to generate the input summary.")
 
            
 st.success(diabetes_diagnosis)
 if selected == 'Heart Disease Prediction':
    st.markdown("<h2 style='text-align:center; color:#FF4B4B;'>❤️ Heart Disease Prediction using ML</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input('Age of the Person')
        cp = st.text_input('Chest pain type (0–3)')
        chol = st.text_input('Serum Cholestoral in mg/dl')
        restecg = st.text_input('Resting Electrocardiographic results (0–2)')
        exang = st.text_input('Exercise Induced Angina (0 or 1)')
        slope = st.text_input('Slope of peak exercise ST segment (0–2)')
        thal = st.text_input('Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)')

    with col2:
        sex = st.text_input('Sex (0 = female, 1 = male)')
        trestbps = st.text_input('Resting Blood Pressure')
        fbs = st.text_input('Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)')
        thalach = st.text_input('Maximum Heart Rate achieved')
        oldpeak = st.text_input('ST depression induced by exercise')
        ca = st.text_input('Number of major vessels colored by fluoroscopy (0–3)')

    heart_diagnosis = ''

    if st.button('Heart Test Result'):
        try:
            input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol),
                          float(fbs), float(restecg), float(thalach), float(exang),
                          float(oldpeak), float(slope), float(ca), float(thal)]

            with st.spinner('Predicting...'):
                heart_prediction = heart_disease_model.predict([input_data])
                confidence = heart_disease_model.predict_proba([input_data])[0][1]

            if heart_prediction[0] == 1:
                heart_diagnosis = f'The person is suffering from Heart Disease. Confidence: {confidence * 100:.2f}%'
                st.error(heart_diagnosis)
            else:
                heart_diagnosis = f'The person is Not suffering from Heart Disease. Confidence: {(1 - confidence) * 100:.2f}%'
                st.success(heart_diagnosis)

        except ValueError:
            st.warning("⚠️ Please enter valid numeric values for all input fields.")

    try:
        if st.button("Show Input Summary - Heart"):
            input_dict = {
                'Age': float(age), 'Sex': float(sex), 'Chest Pain': float(cp),
                'Resting BP': float(trestbps), 'Cholesterol': float(chol),
                'FBS': float(fbs), 'Rest ECG': float(restecg),
                'Max HR': float(thalach), 'Exercise Angina': float(exang),
                'Oldpeak': float(oldpeak), 'Slope': float(slope),
                'CA': float(ca), 'Thal': float(thal)
            }
            input_df = pd.DataFrame(input_dict.items(), columns=["Feature", "Value"])
            st.bar_chart(input_df.set_index("Feature"))
    except ValueError:
        st.warning("Please enter all values to generate the input summary.")
    st.success(heart_diagnosis)
    
    
    

    
#Parkinsons Prediction Page
if selected == 'Parkinsons Prediction':
    st.markdown("<h2 style='text-align:center; color:#FF4B4B;'>🧠 Parkinson's Disease Prediction using ML</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        fhi = st.text_input('MDVP:Fhi(Hz)')
        flo = st.text_input('MDVP:Flo(Hz)')
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        RAP = st.text_input('MDVP:RAP')
        PPQ = st.text_input('MDVP:PPQ')
        DDP = st.text_input('Jitter:DDP')
        Shimmer = st.text_input('MDVP:Shimmer')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        APQ = st.text_input('MDVP:APQ')
        DDA = st.text_input('Shimmer:DDA')
        NHR = st.text_input('NHR')
        HNR = st.text_input('HNR')
        RPDE = st.text_input('RPDE')
        DFA = st.text_input('DFA')
        spread1 = st.text_input('Spread1')
        spread2 = st.text_input('Spread2')
        D2 = st.text_input('D2')
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button('Parkinsons Test Result'):
        try:
            input_data = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                          float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                          float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR),
                          float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]

            with st.spinner('Predicting...'):
                parkinsons_prediction = parkinsons_model.predict([input_data])
                confidence = parkinsons_model.predict_proba([input_data])[0][1]

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = f'The person is suffering from Parkinson\'s Disease. Confidence: {confidence * 100:.2f}%'
                st.error(parkinsons_diagnosis)
            else:
                parkinsons_diagnosis = f'The person is Not suffering from Parkinson\'s Disease. Confidence: {(1 - confidence) * 100:.2f}%'
                st.success(parkinsons_diagnosis)

        except ValueError:
            st.warning("⚠️ Please enter valid numeric values for all input fields.")

    try:
        if st.button("Show Input Summary - Parkinson"):
            input_dict = {
                'Fo': float(fo), 'Fhi': float(fhi), 'Flo': float(flo),
                'Jitter(%)': float(Jitter_percent), 'Jitter(Abs)': float(Jitter_Abs),
                'RAP': float(RAP), 'PPQ': float(PPQ), 'DDP': float(DDP),
                'Shimmer': float(Shimmer), 'Shimmer(dB)': float(Shimmer_dB),
                'APQ3': float(APQ3), 'APQ5': float(APQ5), 'APQ': float(APQ),
                'DDA': float(DDA), 'NHR': float(NHR), 'HNR': float(HNR),
                'RPDE': float(RPDE), 'DFA': float(DFA), 'Spread1': float(spread1),
                'Spread2': float(spread2), 'D2': float(D2), 'PPE': float(PPE)
            }
            input_df = pd.DataFrame(input_dict.items(), columns=["Feature", "Value"])
            st.bar_chart(input_df.set_index("Feature"))
    except ValueError:
        st.warning("Please enter all values to generate the input summary.")

                
    st.success(parkinsons_diagnosis)
        
        