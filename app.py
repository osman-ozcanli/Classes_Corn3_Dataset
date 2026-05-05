import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Page
# ---------------------------
st.set_page_config(page_title="Corn Variety Classification", layout="wide")
st.title("🌽 Corn Variety Classification")
st.caption("Baseline CNN vs VGG16 Transfer Learning")

# ---------------------------
# Paths
# ---------------------------
BASE_MODEL_PATH = "corn_varieties_cnn.keras"
TL_MODEL_PATH   = "corn_vgg16_transfer.keras"

CLASS_NAMES = [
    "Zea mays Chulpi Cancha",
    "Zea mays Indurata",
    "Zea mays Rugosa"
]

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# ---------------------------
# Preprocess
# ---------------------------
def preprocess(img, model_type):
    img = img.convert("RGB")

    if model_type == "base":
        img = img.resize((128, 128))
        arr = np.array(img) / 255.0
    else:
        img = img.resize((170, 170))
        arr = tf.keras.applications.vgg16.preprocess_input(np.array(img))

    return np.expand_dims(arr, axis=0)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Model Selection")

choice = st.sidebar.radio(
    "Choose model:",
    ["Baseline CNN", "VGG16 Transfer Learning"]
)

if choice == "Baseline CNN":
    model = load_model(BASE_MODEL_PATH)
    model_type = "base"
else:
    model = load_model(TL_MODEL_PATH)
    model_type = "tl"

st.sidebar.markdown("**Classes:**")
st.sidebar.write(", ".join(CLASS_NAMES))

# ---------------------------
# Main
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    img_file = st.file_uploader(
        "Upload a corn image",
        type=["jpg", "jpeg", "png"]
    )

    if img_file:
        image = Image.open(img_file)
        st.image(image, use_container_width=True)

with col2:
    st.subheader("Prediction")

    if img_file:
        x = preprocess(image, model_type)
        preds = model.predict(x, verbose=0)[0]

        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        conf = preds[idx] * 100

        st.markdown(f"### 🌽 {label}")
        st.markdown(f"**Confidence:** `{conf:.2f}%`")

    else:
        st.info("Upload an image to see prediction")

st.divider()
st.caption("Corn Variety Classification • Streamlit • Hugging Face")
