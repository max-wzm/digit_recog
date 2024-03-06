import base64
import json
from io import BytesIO
from uuid import uuid4
from handwritten_multi_digit_number_recognition import readutils
from handwritten_multi_digit_number_recognition.recognizer import Recognizer
import google.protobuf
import numpy as np
import requests
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit as st


model = Recognizer()
IMAGE_SCALE_FACTOR = 4


def convert_image_to_b64(canvas_image):
    image_pil = Image.fromarray(canvas_image.astype(np.uint8))
    image_pil = image_pil.convert("L")
    width, height = image_pil.size
    image_pil = image_pil.resize(
        (width // IMAGE_SCALE_FACTOR, height // IMAGE_SCALE_FACTOR)
    )
    image_file = BytesIO()
    image_pil.save(image_file, "png")
    image_file.seek(0)
    image_bytes = image_file.getvalue()
    b64_string = base64.b64encode(image_bytes).decode("utf8")
    image_file.close()
    return b64_string


def get_prediction(b64_string):
    img = readutils.read_b64_image(b64_string, grayscale=True)
    img.save(uuid4().hex+".png")
    pred = model.predict(img)
    return pred


def main():
    st.set_page_config(page_title="口算练习手写识别")
    st.title("口算练习手写识别")

    st.write("画一个三位以内的数字:")
    canvas_result = st_canvas(
        stroke_width=12,
        stroke_color="white",
        background_color="black",
        background_image=None,
        update_streamlit=True,
        height=128,
        width=336,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Submit"):
        if canvas_result.image_data is None:
            st.error("No image data")
        else:
            with st.spinner("Wait for it..."):
                b64_string = convert_image_to_b64(canvas_result.image_data)
                prediction = get_prediction(b64_string)
                st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
