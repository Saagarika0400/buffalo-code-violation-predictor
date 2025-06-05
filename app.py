# app.py

import streamlit as st
import numpy as np
import pandas as pd
import torch
import pickle

# importing my model that I have created
from model_df import CodeViolationNet  

# LOAD MODEL + SCALER + OHE (CACHED)
@st.cache_resource(show_spinner=False)
def load_model_and_artifacts():
    device = torch.device("cpu")

    # 1.1) Load PyTorch model
    model = CodeViolationNet(input_dim=13, hidden1=64, hidden2=32, output_dim=2)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval().to(device)

    # 1.2) Load StandardScaler from my model
    with open("scaler.pkl", "rb") as f_scaler:
        scaler = pickle.load(f_scaler)

    # 1.3) Load OneHotEncoder from my model
    with open("ohe.pkl", "rb") as f_ohe:
        ohe = pickle.load(f_ohe)

    bg_categories = ohe.categories_[0].tolist()

    return model, scaler, ohe, bg_categories

model, scaler, ohe, BG_CATEGORIES = load_model_and_artifacts()

st.set_page_config(
    page_title="Buffalo Code‐Violation Predictor",
    page_icon="📍",
    layout="centered",
)

st.title("🏙️ Buffalo Code‐Violation Closure Predictor")
st.markdown(
    """
    This app predicts whether a Buffalo code‐violation case will be **Closed** (✅)  
    or **Not Closed** (❌).

    Our neural network (PyTorch) attains over **85% accuracy** on held‐out test data.  
    Fill in the details below and click **Predict**.
    """
)
st.markdown("---")

with st.sidebar.form(key="input_form"):
    st.header("Enter Case Details")

    # Month (1–12)
    month = st.selectbox("Month (1–12)", options=list(range(1, 13)), index=0)

    # Day of month (1–31)
    day = st.selectbox("Day of Month (1–31)", options=list(range(1, 32)), index=0)

    # DayOfWeek (0=Mon … 6=Sun) decoded with labels
    # Using a dictionary to map integers to human-readable labels
    dow_map = {
        0: "Monday (0)",
        1: "Tuesday (1)",
        2: "Wednesday (2)",
        3: "Thursday (3)",
        4: "Friday (4)",
        5: "Saturday (5)",
        6: "Sunday (6)",
    }
    dayofweek = st.selectbox(
        "Day of Week",
        options=list(dow_map.keys()),
        format_func=lambda x: dow_map[x],
        index=0,
    )

    # Block Group (exact dtype as used during training)
    blockgroup = st.selectbox(
        "2010 Census Block Group",
        options=BG_CATEGORIES,
        help="Pick one of the block groups exactly as during training (e.g. an int or string)."
    )

    # Latitude
    latitude = st.number_input(
        "Latitude (e.g. 42.9378)",
        value=42.937800,
        format="%.6f",
        step=0.000100,
    )

    # Longitude
    longitude = st.number_input(
        "Longitude (e.g. -78.8658)",
        value=-78.865800,
        format="%.6f",
        step=0.000100,
    )

    st.markdown("---")
    submit_button = st.form_submit_button(label="Predict")


def predict_case(month, day, dayofweek, blockgroup, latitude, longitude):

    # (1) Numeric part: shape (1,5)
    num_arr = np.array([[month, day, dayofweek, latitude, longitude]], dtype=np.float32)

    # (2) Block‐group input as 2D array
    bg_arr = np.array([[blockgroup]])

    # Determine how many categories the encoder expects
    n_cats = ohe.categories_[0].shape[0]

    # (3) Attempt one-hot encoding
    try:
        raw_ohe = ohe.transform(bg_arr)  
    except Exception:
        # If transform fails (e.g. unexpected dtype), fall back to all‐zeros
        raw_ohe = None

    if (
        isinstance(raw_ohe, np.ndarray)
        and raw_ohe.ndim == 2
        and raw_ohe.shape[1] == n_cats
    ):
        bg_ohe = raw_ohe.astype(np.float32)
    else:
        bg_ohe = np.zeros((1, n_cats), dtype=np.float32)

    # Concatenate numeric and one-hot encoded arrays
    # Ensure both arrays are 2D with shape (1, n_features)
    X = np.concatenate([num_arr, bg_ohe], axis=1) 

    # (5) Scale the first 5 columns
    try:
        X[:, 0:5] = scaler.transform(X[:, 0:5])
    except Exception:
        # If scaling fails for some reason, return None so Streamlit shows no crash
        return None

    # Convert to torch tensor and run model
    X_tensor = torch.from_numpy(X).to(torch.device("cpu"))
    with torch.no_grad():
        logits = model(X_tensor)               # shape (1, 2)
        probs  = torch.softmax(logits, dim=1)  # shape (1, 2)
        pred_idx   = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    label_map = {0: "Not Closed ❌", 1: "Closed ✅"}
    return label_map[pred_idx], confidence

if submit_button:
    result = predict_case(
        month=month,
        day=day,
        dayofweek=dayofweek,
        blockgroup=blockgroup,
        latitude=latitude,
        longitude=longitude,
    )
    if result:
        pred_label, conf_score = result

        st.success(f"**Prediction:** {pred_label}")
        st.info(f"**Confidence:** {conf_score*100:.2f}%")
        st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=12)
        st.markdown(
            """
            **How to interpret:**  
            - **Closed (✅)**: Model predicts the case will close on its own.  
            - **Not Closed (❌)**: Flag this location for inspector follow‐up.  
            """
        )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#999; font-size:0.8em;">
    © 2025 Buffalo Code‐Violation Analytics | Built with [Streamlit](https://streamlit.io) & PyTorch | CSE 676 Deep Learning Project | saagarik
    </div>
    """,
    unsafe_allow_html=True,
)
