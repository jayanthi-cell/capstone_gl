
import numpy as np
import streamlit as st
import tensorflow as tf

from preprocessing import (
    preprocess_dicom,
    preprocess_image,
    prepare_model_input,
    get_prediction_label
)


st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="centered"
)


IMG_SIZE = 224
MODEL_PATH = "model/pneumonia_vgg16_savedmodel"


@st.cache_resource
def load_model():
    saved_model = tf.saved_model.load(MODEL_PATH)
    return saved_model.signatures["serving_default"]


predict_fn = load_model()


st.title("🩻 Pneumonia Detection from Chest X-rays")
st.write(
    "Upload a chest X-ray in DICOM, PNG, JPG, or JPEG format. "
    "The model predicts whether pneumonia is present and displays its probability."
)


threshold = st.slider(
    "Classification threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)


uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["dcm", "png", "jpg", "jpeg"]
)


if uploaded_file is not None:
    st.subheader("Uploaded Chest X-ray")

    try:
        if uploaded_file.name.lower().endswith(".dcm"):
            image = preprocess_dicom(uploaded_file)
        else:
            image = preprocess_image(uploaded_file)

        st.image(
            image,
            caption="Uploaded chest X-ray",
            use_container_width=True,
            clamp=True
        )

        display_image, model_input = prepare_model_input(
            image,
            img_size=IMG_SIZE
        )

        prediction = predict_fn(
            tf.constant(model_input, dtype=tf.float32)
        )

        probability = float(list(prediction.values())[0][0][0])

        predicted_class = get_prediction_label(
            probability,
            threshold=threshold
        )

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Class", predicted_class)

        with col2:
            st.metric("Pneumonia Probability", f"{probability:.2%}")

        st.progress(min(max(probability, 0.0), 1.0))

        st.write(
            f"Pneumonia probability: **{probability:.3f}**. "
            f"At threshold **{threshold:.2f}**, prediction is **{predicted_class}**."
        )

        if predicted_class == "Pneumonia":
            st.warning(
                "The model predicts findings consistent with pneumonia. "
                "This is not a medical diagnosis."
            )
        else:
            st.success(
                "The model does not predict pneumonia at the selected threshold. "
                "This is not a medical diagnosis."
            )

    except Exception as e:
        st.error(
            f"Could not process the uploaded file. "
            f"Please upload a valid DICOM, PNG, JPG, or JPEG chest X-ray.\n\nError: {e}"
        )

else:
    st.info("Please upload a DICOM, PNG, JPG, or JPEG chest X-ray file.")


st.divider()
st.caption(
    "For educational and demonstration purposes only. "
    "Not a substitute for professional medical diagnosis."
)
