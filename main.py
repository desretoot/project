from transformers import pipeline
import streamlit as st
from PIL import Image
from io import BytesIO


@st.cache_resource
def load_model():
    task = "image-to-text"
    model = "nlpconnect/vit-gpt2-image-captioning"
    image_to_text = pipeline(task=task, model=model)
    return image_to_text


def check_img(uploaded_file):
    if not uploaded_file:
        raise TypeError('File is missing')

    if not isinstance(uploaded_file, BytesIO):
        raise TypeError('File wrong format')

    try:
        img = Image.open(uploaded_file)
    except Exception as err:
        print(err)
        raise TypeError('File is not image')

    return img


def model_pred(img, model):
    predict = model(img)[0]
    return predict['generated_text']


def load_image(model):
    uploaded_file = st.file_uploader(
        label='Выберите изображение для описания')

    try:
        img = check_img(uploaded_file)
    except Exception as err:
        st.write(str(err))
        return

    st.image(img)

    result = st.button('Распознать изображение')
    if result:
        predict = model_pred(img, model)
        st.write(
            f'**Результаты распознавания: {predict}**')


if __name__ == '__main__':

    st.title('Описание изображений онлайн')
    model = load_model()
    load_image(model)
