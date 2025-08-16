import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import json
import io
import altair as alt
import os

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide",
    page_icon="🧠"
)
st.markdown("""
<style>
.title {
    font-size:40px;
    font-weight:bold;
    color:#8E44AD;
    text-align:center;
    line-height: 1;
}
.prediction {
    font-size:24px;
    font-weight:bold;
    color:#FF5733;
    text-align:center;
}
.footer {
    font-size:14px;
    text-align:center;
    color:#8E44AD;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Brain Tumor Detection</div>", unsafe_allow_html=True)
st.write("")

st.sidebar.header("📌 About")
st.sidebar.info(
    """
    Upload an MRI image and this app will predict the type of brain tumor using a deep learning model.
    
    **Model Classes**:
    - Glioma
    - Meningioma
    - Pituitary
    - No Tumor
    """
)
st.sidebar.markdown("---")
st.sidebar.write("**Developed by:** Eman Fatima")

MODEL_PATH = "brain_tumor_classifier.keras"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 64

@st.cache_resource
def load_my_model(path):
    return load_model(path)


model = load_my_model(MODEL_PATH)

def load_class_labels(json_path=CLASS_INDICES_PATH):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        idx_to_class = {int(v): k for k, v in class_indices.items()}
        labels = [idx_to_class[i] for i in range(len(idx_to_class))]
        return labels
    return ['glioma', 'meningioma', 'pituitary', 'no_tumor']

class_labels = load_class_labels()

def preprocess_image(image_file, img_size=IMG_SIZE):
    img = Image.open(image_file).convert('L')
    img = img.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array
    
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    with col2:
        processed = preprocess_image(uploaded_file)

        if model.input_shape[-1] == 3 and processed.shape[-1] == 1:
            processed = np.repeat(processed, 3, axis=-1)

        with st.spinner("🔍 Analyzing MRI..."):
            preds = model.predict(processed)
        
        probs = preds[0]
        top_idx = int(np.argmax(probs))
        top_label = class_labels[top_idx] if top_idx < len(class_labels) else f"Class {top_idx}"
        top_prob = float(probs[top_idx])

        st.markdown(f"<div class='prediction'>Prediction: {top_label} ({top_prob*100:.2f}%)</div>", unsafe_allow_html=True)
        
        df = pd.DataFrame({"Class": class_labels, "Probability": probs})
        df = df.sort_values("Probability", ascending=False)

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Class', sort=None),
            y='Probability',
            color='Class'
        ).properties(width=500, height=300)
        st.altair_chart(chart, use_container_width=True)
st.markdown("<div class='footer'>⚡ Powered by TensorFlow & Streamlit | UI Enhanced by Eman Fatima</div>", unsafe_allow_html=True)
