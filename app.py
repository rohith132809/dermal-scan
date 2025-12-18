# app.py
import streamlit as st
from PIL import Image
import io
import time
import os

from backend import load_model, detect_and_predict, DEVICE, MODEL_PATH, CLASS_NAMES, AGE_MAP

st.set_page_config(page_title="Facial Condition Detector", layout="centered")

# -------------------------
# UI: header
# -------------------------
st.title("Dermal Scan")
st.write("Upload an image, detect faces, and classify facial condition. Results are annotated and saved to `results/`.")

# -------------------------
# Load model (cached resource)
# -------------------------
@st.cache_resource
def get_model():
    # load_model handles full-model or state_dict as needed
    model = load_model(model_path=MODEL_PATH, num_classes=len(CLASS_NAMES))
    return model

with st.spinner("Loading model..."):
    try:
        model = get_model()
        st.success(f"Model loaded ({MODEL_PATH}). Device: {DEVICE}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# -------------------------
# Upload
# -------------------------
uploaded = st.file_uploader("Upload image (.jpg/.png)", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns([2,1])

if uploaded is not None:
    # display preview
    image = Image.open(uploaded).convert("RGB")
    col1.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze Image"):
        start_time = time.perf_counter()
        with st.spinner("Detecting and classifying..."):
            result = detect_and_predict(model, image)
        elapsed = time.perf_counter() - start_time

        if result["faces"] == []:
            st.warning("No faces detected.")
        else:
            # show annotated image if exists
            annotated_path = result.get("annotated_image_path")
            if annotated_path and os.path.exists(annotated_path):
                st.image(annotated_path, caption="Annotated result", use_column_width=True)
            # show table of faces
            st.subheader("Results")
            for f in result["faces"]:
                label = f["label"]
                conf = f["confidence"] * 100
                bbox = f["box"]
                age_est = AGE_MAP.get(label, "N/A")
                st.markdown(f"**Face {f['face_id']}** — `{label}` — {conf:.2f}%")
                st.write(f"Bounding box: {bbox}")
            st.success(f"Processed in {result['processing_time']:.3f} s (end-to-end). App timer: {elapsed:.3f} s")

        st.info("Annotated images saved in the `results/` folder. Predictions are logged to `predictions.log`.")
else:
    st.info("Upload an image to start inference.")
