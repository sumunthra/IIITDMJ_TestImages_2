import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="Digit Classifier")
st.title("Handwritten Digit Classifier")


@st.cache_resource
def load_model():
    return keras.models.load_model("mnist.keras", compile=False)


model = load_model()

# Show which version this Shipyard environment is running
info = json.load(open("model_info.json"))
st.sidebar.header("Model info")
st.sidebar.write("Version:", os.environ.get("APP_VERSION", "local"))
st.sidebar.write("Model:", info["model"])
st.sidebar.write(f"Test accuracy: {info['test_accuracy']:.2%}")


def clean(image):
    """Grayscale, white digit on black background, faint background removed."""
    gray = np.asarray(image.convert("L"), dtype="float32") / 255.0

    # If the border is bright, the paper is white -> invert (MNIST is white on black)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() > 0.5:
        gray = 1.0 - gray

    # Stretch contrast to 0..1 and remove paper texture / shadows
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
    gray[gray < 0.35] = 0
    return gray


def to_mnist_style(gray):
    """Crop the digit, fit it in a 20x20 box, put it on a 28x28 canvas, centre it."""
    rows = np.where(gray.max(axis=1) > 0)[0]
    cols = np.where(gray.max(axis=0) > 0)[0]
    gray = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = gray.shape
    scale = 20 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray((gray * 255).astype("uint8")).resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(small, dtype="float32") / 255.0

    canvas = np.zeros((28, 28), dtype="float32")
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small

    # Move the centre of mass to the middle, exactly like MNIST
    total = canvas.sum()
    cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
    cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
    canvas = np.roll(canvas, int(round(13.5 - cy)), axis=0)
    canvas = np.roll(canvas, int(round(13.5 - cx)), axis=1)
    return canvas


def split_digits(gray):
    """Split a written number into single digits using empty columns between them."""
    has_ink = gray.max(axis=0) > 0
    pieces, start = [], None
    for x, ink in enumerate(has_ink):
        if ink and start is None:
            start = x
        if not ink and start is not None:
            pieces.append(gray[:, start:x])
            start = None
    if start is not None:
        pieces.append(gray[:, start:])

    # Ignore tiny specks of noise
    biggest = max(p.sum() for p in pieces)
    return [p for p in pieces if p.sum() > 0.1 * biggest]


def true_label(path):
    """Read the true digit from the file name: mnist_7.png -> 7, wrong_4_123.png -> 4."""
    parts = Path(path).stem.split("_")
    if parts[0] in ("mnist", "wrong") and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return None


def predict_digit(image):
    """Clean the image, make it MNIST-style and return (digit, confidence)."""
    gray = clean(image)
    x = to_mnist_style(gray)
    p = model(x[None, ..., None], training=False).numpy()[0]
    return int(p.argmax()), float(p.max())


mode = st.radio("What is in the image?", ["One digit", "A number (several digits)"])

tab_samples, tab_upload, tab_camera = st.tabs(["Test images", "Upload image", "Use camera"])

# Gallery of test images: click "Use" under an image to send it to the model
with tab_samples:
    sample_files = sorted(Path("samples").glob("*.png"))
    st.write("Click **Use** under an image to test it.")
    cols = st.columns(5)
    for i, path in enumerate(sample_files):
        with cols[i % 5]:
            st.image(str(path), caption=path.stem, width=100)
            if st.button("Use", key=path.name):
                st.session_state["sample"] = str(path)

    # Run the model on every labelled test image and show a score
    if st.button("Test all images"):
        rows, correct = [], 0
        for path in sample_files:
            label = true_label(path)
            if label is None:          # skip images without a label (e.g. number_2024.png)
                continue
            digit, conf = predict_digit(Image.open(path))
            correct += int(digit == label)
            rows.append({
                "image": path.name,
                "true": label,
                "predicted": digit,
                "confidence": f"{conf:.0%}",
                "result": "✅" if digit == label else "❌",
            })
        st.metric("Gallery accuracy", f"{correct}/{len(rows)}")
        st.dataframe(rows, hide_index=True)

with tab_upload:
    uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
with tab_camera:
    photo = st.camera_input("Take a photo of a digit written on paper")

# Use the uploaded file or camera photo if there is one, otherwise the clicked test image
source = uploaded or photo or st.session_state.get("sample")

if source:
    image = Image.open(source)
    image.thumbnail((400, 400))
    st.image(image, caption="Your image", width=200)

    gray = clean(image)
    if gray.max() == 0:
        st.error("No digit found in the image.")
        st.stop()

    if mode == "One digit":
        pieces = [gray]
    else:
        pieces = split_digits(gray)

    inputs = np.stack([to_mnist_style(p) for p in pieces])
    probs = model(inputs[..., None], training=False).numpy()
    digits = probs.argmax(axis=1)

    st.success("Prediction: " + "".join(str(d) for d in digits))

    # For gallery images we know the true digit, so show if the model is right
    if isinstance(source, str) and mode == "One digit" and true_label(source) is not None:
        label = true_label(source)
        if digits[0] == label:
            st.write(f"✅ Correct (true digit: {label})")
        else:
            st.write(f"❌ Wrong (true digit: {label})")

    st.write("What the model sees:")
    columns = st.columns(len(pieces))
    for i, col in enumerate(columns):
        col.image(inputs[i], caption=f"{digits[i]} ({probs[i].max():.0%})", width=80, clamp=True)

    if mode == "One digit":
        st.write("Probability of each digit:")
        st.bar_chart(probs[0])

    if probs.max(axis=1).min() < 0.7:
        st.warning("Low confidence. Try a darker pen, a plain background, or a tighter crop.")
