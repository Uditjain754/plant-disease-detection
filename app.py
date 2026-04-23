import streamlit as st
import torch
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from train import PlantDiseaseModel, predict_image
import warnings

warnings.filterwarnings("ignore")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model_resources():
    import json

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model, transform, label_encoder, class_names, device


# ---------------- PREDICT ---------------- #
def predict(image_file, model, transform, label_encoder, device):
    with open("temp.jpg", "wb") as f:
        f.write(image_file.getvalue())

    class_name, confidence, probs = predict_image(
        model, "temp.jpg", transform, device, label_encoder
    )

    top_idx = np.argsort(probs)[::-1][:5]
    top_classes = [label_encoder.inverse_transform([i])[0] for i in top_idx]
    top_probs = [probs[i] * 100 for i in top_idx]

    os.remove("temp.jpg")

    return class_name.replace("_", " "), confidence, top_classes, top_probs


# ---------------- MAIN UI ---------------- #
def main():
    st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

    # ----------- STYLE ----------- #
    st.markdown("""
    <style>
    body {
        background-color: #f5f7f6;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #2e7d32;
    }

    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* REMOVE CURSOR + INTERACTION */
    [data-testid="stTable"] {
        pointer-events: none;
        user-select: none;
    }

    [data-testid="stTable"] * {
        caret-color: transparent !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # ----------- HEADER ----------- #
    st.markdown("<div class='title'>🌿 Plant Disease Classifier</div>", unsafe_allow_html=True)
    st.write("Upload a plant leaf image to detect disease")

    # ----------- LOAD MODEL ----------- #
    try:
        model, transform, label_encoder, class_names, device = load_model_resources()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # ----------- LAYOUT ----------- #
    left, right = st.columns([1, 2])

    # ----------- LEFT PANEL ----------- #
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Upload Image")

        uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])

        st.markdown("</div>", unsafe_allow_html=True)

        # ----------- DROPDOWN MENU ----------- #
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        with st.expander("📋 Available Diseases", expanded=False):
            formatted = [c.replace("_", " ").title() for c in class_names]

            df = pd.DataFrame({
                "Available Diseases": formatted
            })

            st.table(df)

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------- RIGHT PANEL ----------- #
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Image Preview")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            # Prediction
            class_name, confidence, top_classes, top_probs = predict(
                uploaded_file, model, transform, label_encoder, device
            )

            st.markdown(f"## Diagnosis: {class_name}")
            st.write(f"Confidence: {confidence:.2f}%")

            # Chart
            df = pd.DataFrame({
                "Disease": [c.replace("_", " ") for c in top_classes],
                "Confidence": top_probs
            })

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(df["Disease"], df["Confidence"])
            ax.set_xlabel("Confidence (%)")
            st.pyplot(fig)

        else:
            st.info("Upload an image to see prediction")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------- FOOTER ----------- #
    st.markdown("---")
    st.markdown("© 2025 Plant Disease Classifier | Developed by **Udit Jain**")


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()