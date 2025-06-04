import joblib
import streamlit as st
import numpy as np


model = joblib.load('deploy/Iris dataset predictive model.pkl')
st.title('Iris Flower Predictor')

sepal_length = st.number_input('Sepal Length')
sepal_width = st.number_input('Sepal Width')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal Width')


if st.button('Predict'):
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    probabilities = model.predict_proba(user_input)

    species = ['Setosa', 'Versicolor', 'Virginica' ]

    for i in range(3):
        st.write(f'{species[i]}: {probabilities [0][i] * 100} ')
    prediction = model.predict(user_input)
    st.success(f'The predicted specie is {species[prediction[0]]}')
