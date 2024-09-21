import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load the model and data
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
pre = pd.read_csv('C:\\Users\\ucino\\sih project\\pridiction_model\\dataset\\precautions_df.csv')
medic = pd.read_csv('C:\\Users\\ucino\\sih project\\pridiction_model\\dataset\\medications.csv')
diet = pd.read_csv('C:\\Users\\ucino\\sih project\\pridiction_model\\dataset\\diets.csv')
workout = pd.read_csv('C:\\Users\\ucino\\sih project\\pridiction_model\\dataset\\workout_df.csv')
discription = pd.read_csv('C:\\Users\\ucino\\sih project\\pridiction_model\\dataset\\description.csv')

symptoms = pickle.load(open('symptoms.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))

def symptom_to_binary(user_symptoms):
    feature = np.zeros(len(symptoms))
    for i in user_symptoms:
        if i in symptoms:
            feature[symptoms.index(i)] = 1
    return feature

def predict_disease(user_symptoms):
    symp = symptom_to_binary(user_symptoms).reshape(1, -1)
    prediction = model.predict(symp)
    disease = le.inverse_transform(prediction)
    return disease[0]

def remidies(disease):
    
    desc = discription[discription['Disease'] == disease]['Description']
    desc = " ".join([w for w in desc])

    # Get precautions
    prec = pre[pre['Disease'] == disease].iloc[0].tolist()
    
    # Get medications
    med = medic[medic['Disease'] == disease]['Medication'].tolist()
    
    # Get diet
    die = diet[diet['Disease'] == disease]['Diet'].tolist()
    
    # Get workout
    wrkout = workout[workout['disease'] == disease]['workout'].tolist()

    return desc, med, die, wrkout, prec

st.title("Disease Prediction App")

user_symptoms = st.multiselect(
    "Select Symptoms:",
    options=symptoms,
    default=[]  
)

if st.button("Predict Disease"):
    if not user_symptoms:
        st.error("Please select at least one symptom.")
    else:
        Disease = predict_disease(user_symptoms)
        desc, med, die, wrkout, prec = remidies(Disease)
        
        st.subheader("Results:")
        st.write(f"**Disease:** {Disease}")
        st.write(f"**Description:** {desc}")
        
        st.write("**Precautions:**")
        if prec:
            for item in prec[1:]:
                st.write(f"- {item}")
        else:
            st.write("No precautions found.")
        
        st.write("**Diet:**")
        if die:
            for item in die:
                 st.write(f"- {item}")
        else:
            st.write("No diet recommendations found.")
        
        st.write("**Medication:**")
        if med:
            for item in med:
                st.write(f"- {item}")
        else:
            st.write("No medication recommendations found.")
        
        st.write("**Workout:**")
        if wrkout:
            for item in wrkout:
                st.write(f"- {item}")
        else:
            st.write("No workout recommendations found.")


