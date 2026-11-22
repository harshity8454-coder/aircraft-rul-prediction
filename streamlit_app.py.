
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Aircraft RUL Prediction", page_icon="✈️")

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "optimized_random_forest.pkl")
    return joblib.load(path)

model = load_model()

st.title("✈️ Aircraft Engine RUL Prediction")
st.write("Remaining Useful Life Prediction using Machine Learning")
st.info("Enter the nine sensor readings below.")

features = [
    "Sensor 11", "Sensor 15", "Sensor 21",
    "Sensor 20", "Sensor 17", "Sensor 2",
    "Sensor 3", "Sensor 8", "Sensor 9"
]

defaults = [
    518.67, 642.00, 1589.70,
    14.62, 392.00, 641.82,
    1589.70, 2388.06, 9046.19
]

readings = []

for feature, default in zip(features, defaults):
    value = st.number_input(feature, value=float(default))
    readings.append(value)

if st.button("Predict Remaining Useful Life"):
    prediction = model.predict(
        np.array(readings).reshape(1, -1)
    )[0]

    prediction = float(np.clip(prediction, 0, 125))

    st.metric("Predicted RUL", f"{prediction:.2f} cycles")

    if prediction <= 20:
        st.error("Critical: Immediate inspection recommended")
    elif prediction <= 50:
        st.warning("Warning: Maintenance planning recommended")
    else:
        st.success("Normal: Continue monitoring")

    st.caption("Educational demonstration only. Not a validated aircraft safety system.")

