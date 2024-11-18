import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
pathlib.PosixPath = pathlib.Path

# Path muvofiqligini sozlash
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath if pathlib.Path().drive else temp

# Sahifa sozlamalari
st.set_page_config(page_title="Rasmlar tanish dasturi")
st.title("Rasmlarni tanish dasturi")
st.write("Klasslar: avtomobil, samolyot, qayiq, yirtqich hayvonlar, musiqa asbobi, sport jihozlari, telefon, ofis jihozlari, oshxona anjomlari")

# Fayl yuklash
files = st.file_uploader("Rasm yuklash", type=["png", "jpeg", "jpg"])

# Rasmni joylash
files = st.file_uploader("Rasm yuklash", type=["avif", "png", "jpeg", "gif", "svg"])
if files:
    st.image(files)  # rasmni chiqarish
    # PIL convert
    img = PILImage.create(files)
    
    # Modelni yuklash
    model = load_learner('transport_model.pkl')

    # Bashorat qiymatni topamiz
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")

    # Plotting
    fig = px.bar(x=probs * 100, y=model.dls.vocab)
    st.plotly_chart(fig)