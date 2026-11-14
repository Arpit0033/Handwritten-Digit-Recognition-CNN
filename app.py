
# ==========================================
# HANDWRITTEN DIGIT RECOGNITION
# IMPROVED STREAMLIT APPLICATION
# ==========================================

# STEP 1: IMPORT LIBRARIES

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path


# ==========================================
# STEP 2: PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


# ==========================================
# STEP 3: CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #4F8BF9;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 17px;
}

.prediction-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #E8F0FE;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# STEP 4: HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '🔢 Handwritten Digit Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Powered by Convolutional Neural Network (CNN)'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Draw any digit from 0 to 9 using your mouse "
    "and let the CNN model recognize it."
)


# ==========================================
# STEP 5: LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    base_dir = Path(__file__).resolve().parent

    model_path = (
        base_dir / "models" / "digit_cnn_model.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found in models folder!"
        )

    return tf.keras.models.load_model(str(model_path))


try:

    model = load_model()

except Exception as error:

    st.error(f"Model loading error: {error}")
    st.stop()


# ==========================================
# STEP 6: IMAGE PREPROCESSING
# ==========================================


def preprocess_image(canvas_data):

    # Convert RGB image into grayscale
    rgb = canvas_data[:, :, :3].astype(np.float32)
    gray = np.mean(rgb, axis=2)

    # Find white digit pixels
    coords = np.argwhere(gray > 30)

    if len(coords) == 0:
        return None

    # Find digit boundaries
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop digit
    digit = gray[y_min:y_max + 1, x_min:x_max + 1]

    # Convert to PIL image
    digit_img = Image.fromarray(
        digit.astype(np.uint8)
    )

    # Resize while maintaining aspect ratio
    digit_img.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create black 28x28 image
    final_img = Image.new("L", (28, 28), 0)

    # Initially place digit in center
    x_offset = (28 - digit_img.width) // 2
    y_offset = (28 - digit_img.height) // 2

    final_img.paste(digit_img, (x_offset, y_offset))

    # Calculate center of mass
    image_array = np.array(final_img).astype(np.float32)

    total_mass = image_array.sum()

    if total_mass == 0:
        return None

    y_coords, x_coords = np.indices((28, 28))

    center_x = (x_coords * image_array).sum() / total_mass
    center_y = (y_coords * image_array).sum() / total_mass

    # Shift digit towards center of mass target
    shift_x = int(round(13.5 - center_x))
    shift_y = int(round(13.5 - center_y))

    # Create shifted image without wrapping pixels
    shifted = Image.new("L", (28, 28), 0)

    shifted.paste(
        final_img,
        (shift_x, shift_y)
    )

    # Normalize
    image_array = np.array(
        shifted
    ).astype("float32") / 255.0

    # CNN input shape
    image_array = image_array.reshape(1, 28, 28, 1)

    return shifted, image_array



# ==========================================
# STEP 7: CANVAS SESSION STATE
# ==========================================

if "canvas_version" not in st.session_state:

    st.session_state.canvas_version = 0


# ==========================================
# STEP 8: MAIN APPLICATION LAYOUT
# ==========================================

left_col, right_col = st.columns(2)


# ==========================================
# LEFT SIDE: DRAWING CANVAS
# ==========================================

with left_col:

    st.subheader("✍️ Draw Your Digit")

    # Clear canvas button
    if st.button(
        "🗑️ Clear Canvas",
        use_container_width=True
    ):

        st.session_state.canvas_version += 1

        st.rerun()

    canvas_result = st_canvas(

        fill_color="rgba(255,255,255,0)",

        stroke_width=18,

        stroke_color="#FFFFFF",

        background_color="#000000",

        width=280,

        height=280,

        drawing_mode="freedraw",

        key=f"digit_canvas_{st.session_state.canvas_version}",

        update_streamlit=True,

        return_image_data=True

    )

    st.caption(
        "Draw a single digit from 0 to 9."
    )

    st.caption(
        "Use the Clear Canvas button to draw again."
    )


# ==========================================
# RIGHT SIDE: PREDICTION RESULT
# ==========================================

with right_col:

    st.subheader("🎯 Prediction Result")

    if canvas_result.image_data is not None:

        processed = preprocess_image(
            canvas_result.image_data
        )

        if processed is not None:

            final_img, input_array = processed

            # ==================================
            # MODEL PREDICTION
            # ==================================

            predictions = model.predict(
                input_array,
                verbose=0
            )[0]

            predicted_digit = int(
                np.argmax(predictions)
            )

            confidence = float(
                np.max(predictions) * 100
            )

            # ==================================
            # DISPLAY PREDICTED DIGIT
            # ==================================

            st.markdown(
                f"""
                <div class="prediction-box">

                    <h1 style="color:#2563EB;
                    font-size:65px;">

                        {predicted_digit}

                    </h1>

                    <h3>Predicted Digit</h3>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ==================================
            # CONFIDENCE INDICATOR
            # ==================================

            st.subheader("📊 Prediction Confidence")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                confidence / 100
            )

            # Confidence warnings
            if confidence < 60:

                st.warning(
                    "Low confidence! Please draw the "
                    "digit more clearly."
                )

            elif confidence < 80:

                st.info(
                    "Moderate confidence. Try drawing "
                    "the digit more clearly."
                )

            else:

                st.success(
                    "The model is confident about "
                    "this prediction."
                )

            # ==================================
            # PROCESSED IMAGE
            # ==================================

            st.subheader("🖼️ Processed 28×28 Image")

            enlarged_img = final_img.resize(
                (140, 140),
                Image.Resampling.NEAREST
            )

            st.image(
                enlarged_img,
                width=140
            )

            # ==================================
            # DIGIT PROBABILITIES
            # ==================================

            st.subheader("📈 Digit Probabilities")

            for digit in range(10):

                probability = float(
                    predictions[digit]
                )

                st.write(
                    f"Digit {digit}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

        else:

            st.warning(
                "Please draw a digit on the canvas."
            )

    else:

        st.info(
            "Your prediction will appear here."
        )


# ==========================================
# STEP 9: FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">

        Handwritten Digit Recognition Project

        <br>

        Built using Python | TensorFlow | CNN | MNIST

    </div>
    """,
    unsafe_allow_html=True
)
