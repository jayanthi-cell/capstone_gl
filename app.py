
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import (
    Input,
    Lambda,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.models import Model, load_model

from preprocessing import (
    preprocess_dicom,
    preprocess_image,
    prepare_model_input,
    get_prediction_label,
)

IMG_SIZE = 224

MODEL_PATH = "pneumonia_detection_resnet50_v1_0.keras"


@st.cache_resource
def get_model():
    # Rebuild exactly the same architecture used during training
    resnet_base = ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )

    for layer in resnet_base.layers:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 1), name="grayscale_input")

    x = RandomRotation(0.05)(inputs)
    x = RandomTranslation(0.1, 0.1)(x)
    x = RandomZoom(0.1)(x)

    x = Lambda(
        lambda z: tf.image.grayscale_to_rgb(z),
        output_shape=(224, 224, 3),
        name="grayscale_to_rgb",
    )(x)

    x = Lambda(
        lambda z: preprocess_input(z),
        output_shape=(224, 224, 3),
        name="resnet_preprocess",
    )(x)

    x = resnet_base(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="ResNet50_Improved")

    # Load only the trained weights; do not deserialize Lambda layers
    model.load_weights(MODEL_PATH)

    return model


model = get_model()

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
    step=0.01,
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["dcm", "png", "jpg", "jpeg"],
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
            clamp=True,
        )

        display_image, model_input = prepare_model_input(
            image,
            img_size=IMG_SIZE,
        )

        prediction = model.predict(model_input, verbose=0)
        probability = float(prediction[0][0])

        predicted_class = get_prediction_label(
            probability,
            threshold=threshold,
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
            "Could not process the uploaded file. "
            "Please upload a valid DICOM, PNG, JPG, or JPEG chest X-ray.\n\n"
            f"Error: {e}"
        )

else:
    st.info("Please upload a DICOM, PNG, JPG, or JPEG chest X-ray file.")

st.divider()
st.caption(
    "For educational and demonstration purposes only. "
    "Not a substitute for professional medical diagnosis."
)
