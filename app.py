import streamlit as st
import numpy as np
import pickle

# Load model & scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

st.set_page_config(page_title="Crop Recommendation", layout="wide")

st.markdown("<h1 style='text-align: center; color: green;'>🌱 Crop Recommendation</h1>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen", value=90.0)
    temp = st.number_input("Temperature (°C)", value=28.0)
    rainfall = st.number_input("Rainfall (mm)", value=150.0)

with col2:
    P = st.number_input("Phosphorus", value=45.0)
    K = st.number_input("Potassium", value=45.0)
    humidity = st.number_input("Humidity (%)", value=80.0)
    ph = st.number_input("pH", value=6.5)

# Button center
c1, c2, c3 = st.columns([1,2,1])
with c2:
    predict_btn = st.button("Get Recommendation")

# Prediction
if predict_btn:
    try:
        # ✅ Ensure float conversion
        feature_list = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]
        single_pred = np.array(feature_list).reshape(1, -1)

        st.write("🔍 Input Features:", feature_list)

        # ✅ Scaling
        scaled = ms.transform(single_pred)
        final = sc.transform(scaled)

        st.write("🔍 After Scaling:", final)

        prediction = model.predict(final)
        st.write("🔍 Raw Prediction:", prediction)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        st.markdown("---")

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            st.success(f"🌾 Recommended Crop
