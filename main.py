from transformers import pipeline
import streamlit as st
from PIL import Image

@st.cache_resource
def load_model():
    task = "image-to-text"
    model = "nlpconnect/vit-gpt2-image-captioning"
    image_to_text = pipeline(task = task, model = model)
    return image_to_text


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для описания')
    if uploaded_file is not None:
        st.image(uploaded_file)
        return Image.open(uploaded_file)
    else:
        return None


st.title('Описание изображений в Streamlit')

img = load_image()
model = load_model()

result = st.button('Распознать изображение')
if result:
    x = model(img)[0]
    for _, v in x.items():
        st.write(f'**Результаты распознавания: {v}**')
