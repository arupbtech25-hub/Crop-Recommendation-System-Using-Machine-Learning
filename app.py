import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except:
    st.error("❌ Model or scaler files not found!")
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crop Recommendation", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: green;'>🌱 Crop Recommendation</h1>",
    unsafe_allow_html=True
)

st.markdown("### Enter Soil & Weather Details")

# ---------------- INPUT LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen", min_value=0.0, value=90.0)
    temp = st.number_input("Temperature (°C)", min_value=0.0, value=28.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=150.0)

with col2:
    P = st.number_input("Phosphorus", min_value=0.0, value=45.0)
    K = st.number_input("Potassium", min_value=0.0, value=45.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, value=80.0)
    ph = st.number_input("pH", min_value=0.0, value=6.5)

# ---------------- BUTTON ----------------
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    predict_btn = st.button("Get Recommendation")

# ---------------- PREDICTION ----------------
if predict_btn:
    try:
        # Feature order MUST match training
        features = np.array([
            float(N), float(P), float(K),
            float(temp), float(humidity),
            float(ph), float(rainfall)
        ]).reshape(1, -1)

        # Scaling
        scaled = ms.transform(features)
        final = sc.transform(scaled)

        # Prediction
        prediction = model.predict(final)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        st.markdown("---")

        pred_value = int(prediction[0])

        if pred_value in crop_dict:
            crop = crop_dict[pred_value]
            st.success(f"🌾 Recommended Crop: {crop}")
            st.balloons()
        else:
            st.error("❌ Could not determine crop")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
