import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

# Ensure pathlib compatibility
pathlib.PosixPath = pathlib.Path

# Set up the app
st.set_page_config(page_title="Rasmlar tanish dasturi")
st.title("Rasmlarni tanish dasturi")
st.write("Klasslar: avtomobil, samolyot, qayiq, yirtqich hayvonlar, musiqa asbobi, sport jihozlari, telefon, ofis jihozlari, oshxona anjomlari")

# File uploader
files = st.file_uploader("Rasm yuklash", type=["avif", "png", "jpeg", "gif", "svg"])

if files:
    # Display image
    st.image(files)
    
    # Convert file to PILImage
    img = PILImage.create(files.getvalue())
    
    # Load the model
    model_path = Path("modelalibek.pkl")
    try:
        model = load_learner(model_path)
        # Make prediction
        pred, pred_id, probs = model.predict(img)
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")
        
        # Plot probabilities
        fig = px.bar(x=probs * 100, y=model.dls.vocab, orientation="h", title="Bashorat ehtimolligi")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Modelni yuklashda xatolik: {e}")

# Sidebar
st.sidebar.header("Qo'shimcha ma'lumotlar")
st.sidebar.write("Bizni ijtimoiy tarmoqlarda kuzatib boring:")
st.sidebar.markdown("[Telegram](https://t.me/ali_bek_003)")
st.sidebar.markdown("[Instagram](https://www.instagram.com/alib_ek0311/profilecard/?igsh=MWo5azN2MmM2cGs0aw==)")
st.sidebar.markdown("[Github](https://github.com/AlibekSerikbayev)")
st.write("Ushbu dastur Alibek Serikbayev tomonidan yaratildi")
