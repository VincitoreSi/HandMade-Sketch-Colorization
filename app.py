'''
file: stream.py
author: Vincit0re
objective: Make a streamlit app to colorize a grayscale sketch image using a pre-trained model.
date: 2023-03-14
'''
from dependencies import *
from hyperparameters import Hyperparameters
from utils import *
from sketch import SketchColor

# Set page title and favicon
st.set_page_config(page_title="Sketch Colorization", page_icon=":art:")

# Set app header and subheader
st.header("Handmade Sketch Colorization Using Segmentation Techniques")
st.subheader(
    "Upload a grayscale sketch image and enter the name of the image to colorize it.")
st.write("<br>", unsafe_allow_html=True)
st.write("----------------------------------------------------------------------------------")

with st.container():
    st.subheader("Upload Section")
    image = st.file_uploader(
        "Upload a grayscale sketch image", type=["png", "jpg", "jpeg"])

    image_name = st.text_input("Enter the name of the image", "sketch")
    st.write("<br>", unsafe_allow_html=True)
    st.write(
        "----------------------------------------------------------------------------------")

with st.container():
    st.subheader("Hyperparameters Section")
    thresh = st.slider("Threshold", 0, 255, 150)
    thresh_type = st.selectbox("Threshold Type", [
        "TOZERO_INV", "BINARY", "BINARY_INV", "TRUNC", "TOZERO"], index=0)
    color_map = st.selectbox(
        "Color Map", Hyperparameters._COLOR_MAPS.keys(), index=7)

    st.write("<br>", unsafe_allow_html=True)
    st.write(
        "----------------------------------------------------------------------------------")

with st.container():
    st.subheader("Output Section")
    if image is not None and image_name != "sketch":
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        sketch_image = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
        # st.image(image, caption='Uploaded Image', use_column_width=True)
        median_val = np.median(sketch_image)/255

        if thresh == 150:
            if median_val < 0.3:
                thresh = 100
            elif median_val < 0.5:
                thresh = 120
            elif median_val < 0.7:
                thresh = 150
            elif median_val < 0.9:
                thresh = 180
            elif median_val < 0.95:
                thresh = 190
            else:
                thresh = 200

        start_time = time.time()

        colorized_sketch = SketchColor(
            image=sketch_image, image_name=image_name, thresh=thresh, thresh_type=thresh_type, save=False, show=False, color_map=color_map).colorization()

        end_time = time.time()

        # show the colorized image
        # st.image(colorized_sketch, caption=f"Colorized image in {end_time-start_time:.2f} seconds", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        with col2:
            st.image(colorized_sketch,
                     caption=f"Colorized image in {end_time-start_time:.2f} seconds", use_column_width=True)

        colorized_bytes = cv.imencode('.jpg', colorized_sketch)[1].tobytes()
        st.write("<br>", unsafe_allow_html=True)
        st.write(
            "----------------------------------------------------------------------------------")
        # Download button for the colorized image
        st.download_button(
            label="Download Colorized Image",
            data=colorized_bytes,
            file_name=f"{image_name}_colorized.jpg",
            mime="image/jpeg"
        )
        st.write(
            "----------------------------------------------------------------------------------")
