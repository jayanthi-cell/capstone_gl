
import io
import cv2
import numpy as np
import pydicom
from PIL import Image

IMG_SIZE = 224


def preprocess_dicom(uploaded_file):
    """
    Read a DICOM chest X-ray and return a grayscale uint8 NumPy array.
    """
    ds = pydicom.dcmread(uploaded_file)
    img = ds.pixel_array

    # Invert MONOCHROME1 images so the appearance matches training data
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    return img.astype(np.uint8)


def preprocess_image(uploaded_file):
    """
    Read a PNG/JPG/JPEG image and convert it to grayscale uint8 NumPy array.
    """
    file_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")

    return np.array(pil_img).astype(np.uint8)


def prepare_model_input(img, img_size=IMG_SIZE):
    """
    Resize an image and convert it to the model input shape:
    (1, img_size, img_size, 1)
    """
    img = cv2.resize(
        img,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    model_input = np.expand_dims(img, axis=-1)      # (224, 224, 1)
    model_input = np.expand_dims(model_input, axis=0)  # (1, 224, 224, 1)

    return img, model_input


def get_prediction_label(probability, threshold=0.5):
    """
    Convert model probability into a class label.
    """
    return "Pneumonia" if probability >= threshold else "No Pneumonia"
