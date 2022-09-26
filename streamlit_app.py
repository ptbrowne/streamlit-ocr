import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
import sys

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def process(t):
    lines = '\n'.join([re.sub('[0-9],?', '', l) for l in t.split('\n') if l != ''])
    return lines

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(layout="wide")
# Upload an image and set some options for demo purposes

img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])

print(img_file.read())
realtime_update = True
box_color = '#0000FF'
aspect_choice = 'Free'

c1, c2, c3 = st.columns(3)

lum = st.sidebar.radio(label="Luminosity", options=[0, 32, 64, 92, 100])
contrast = st.sidebar.radio(label="Contrast", options=[0, 32, 64, 92])

lang = st.sidebar.radio(label="Language", options=["French", "German"])
lang = ({'French': 'fra', 'German': 'deu'})[lang]
aspect_dict = {
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

c1, c2 = st.columns(2)

if img_file:
    pilimg = Image.open(img_file)
    cvimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cvimg = apply_brightness_contrast(cvimg, lum, contrast)
    pilimg = Image.fromarray(cv2.cvtColor(np.array(cvimg), cv2.COLOR_BGR2RGB))
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    with c1:
        cropped_img = st_cropper(pilimg, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    text = pytesseract.image_to_string(cropped_img, lang=lang)

    with c2:
        result = process(text)
        st.text_area('Result', value=result, height=26 * len(result.split('\n')))
