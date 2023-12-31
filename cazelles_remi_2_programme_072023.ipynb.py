#***************************************************************
import streamlit as st
import subprocess
import time

with st.spinner('Installing dependencies...'):
    subprocess.run(["streamlit", "cache", "clear"])
    time.sleep(0.05)
    subprocess.run(["pip", "install", "--upgrade", "pip"], check=True, capture_output=True)

    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    st.success("Requirements satisfied")

#***************************************************************

import pickle
import numpy as np

st.success("apres pickle et numpy")

from PIL import Image


st.success("apres PIL")

import tensorflow as tf

st.success("apres tf")


from tensorflow.keras.models import load_model

st.success("librairies imported")

def get_model_and_breed_names():
    """ Load model and breed_names"""
    model = load_model("./model/model_vgg16_fine_tuned_6pic_4transf.keras", compile=False)
    breed_names = pickle.load(open("./model/breed_names.save", "rb"))
    return model, class_names

st.success("model loaded")

def get_image(img):
    """ transform into array and preprocess image """
    img_array = Image.fromarray(img, 'RGB')
    resized_img = img_array.resize((224, 224))
    image = np.array(resized_img)
    image = image.astype(np.float32)
    image = image/255
    image = np.expand_dims(image, axis=0)
    return img_array

def predict_breed(model, img, labels_):
    """ Make prediction using model """
    pred = model.predict(img)
    label = labels_[np.argmax(pred)]
    return label



def main():
    st.title("Cette API donne la race d'un chien présent sur une photo")
    file = st.file_uploader("Veuillez charger une image")
    img_placeholder = st.empty()
    success = st.empty()
    submit_placeholder = st.empty()
    submit=False

    if file is not None:
        with st.spinner("hargement de l'image.."):
            model, breed_names = get_model_and_breed_names()
            img = Image.open(file)
            img_placeholder.image(img, width=299)
        submit = submit_placeholder.button("Lancer la détection de race")

    if submit:
        with st.spinner('Résultat en attente...'):
            submit_placeholder.empty()
            img_tensor = get_image(img)
            pred = predict_breed(model=model, img=img_tensor, labels_=breed_names)
            success.success("Pour le modèle il s'agit d'un {}".format(pred))

if __name__ == "__main__":
    main()
