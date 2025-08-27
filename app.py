import streamlit as st
import pandas as pd
from sklearn.externals import joblib

model=joblib.load("student_pass_model.pkl")
st.title("Student pass prediction")
st.write("Enter the scores below to predict the student will pass or fail")


math=st.number_input("Math score",min_value=0,max_value=100,step=1)
reading=st.number_input("Reading score",min_value=0,max_value=100,step=1)
writing=st.number_input("Writing score",min_value=0,max_value=100,step=1)

if st.button("Predict"):
    input_data=pd.DataFrame([[math,reading,writing]],
                            columns=['math score','reading score','writing score'])
    prediction=model.predict(input_data)[0]
    probability=model.predict_proba(input_data)[0]
    if prediction==1:
       label='pass'
       confidence=probability[1]
    else:
        label='fail'
        confidence=probability[0]
    st.write(f"The student is predicted to {label} with {confidence*100:.2f}% confidence")
