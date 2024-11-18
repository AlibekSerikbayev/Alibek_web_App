import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
pathlib.PosixPath = pathlib.Path

# Ilovani ishga tushirish uchun ko'rsatmalar
st.set_page_config(page_title="Rasmlar tanish dasturi")
st.title("Rasmlarni tanish dasturi")
st.write("Klasslar avtomobil samolyoti qayiq Yirtqich hayvonlar musiqa asbobi Sport jihozlari Telefon ofis jihozlari oshxona anjomlari")


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


    # Ijtimoiy tarmoq va GitHub sahifalarini ko'rsatish (Display social media and GitHub links)
st.sidebar.header("Qo'shimcha ma'lumotlar")
st.sidebar.write("Bizni ijtimoiy tarmoqlarda kuzatib boring:")
st.sidebar.markdown("[Telegram](https://t.me/ali_bek_003)")
st.sidebar.markdown("[Instagram](https://www.instagram.com/alib_ek0311/profilecard/?igsh=MWo5azN2MmM2cGs0aw==)")
st.sidebar.markdown("[Github](https://github.com/AlibekSerikbayev)")
st.write("Ushbu dastur Alibek Serikbayev tomonidan yaratildi")