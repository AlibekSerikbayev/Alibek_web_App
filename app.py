import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
from pathlib import Path

# Path muvofiqligini sozlash
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath if pathlib.Path().drive else temp

# Sahifa sozlamalari
st.set_page_config(page_title="Rasmlar tanish dasturi")
st.title("Rasmlarni tanish dasturi")
st.write("Klasslar: avtomobil, samolyot, qayiq, yirtqich hayvonlar, musiqa asbobi, sport jihozlari, telefon, ofis jihozlari, oshxona anjomlari")

# Fayl yuklash
files = st.file_uploader("Rasm yuklash", type=["png", "jpeg", "jpg"])

if files:
    # Rasmni ko'rsatish
    st.image(files, caption="Yuklangan rasm")
    
    # Rasmni PILImage formatiga o'tkazish
    img = PILImage.create(files.getvalue())
    
    # Modelni yuklash
    try:
        model_path = Path("modelalibek.pkl")
        model = load_learner(model_path)
        
        # Bashorat qilish
        pred, pred_id, probs = model.predict(img)
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")
        
        # Diagramma yaratish
        fig = px.bar(x=probs * 100, y=model.dls.vocab, orientation='h', title="Bashorat ehtimolligi")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Modelni yuklashda xatolik: {e}")

# Sidebar
st.sidebar.header("Qo'shimcha ma'lumotlar")
st.sidebar.markdown("[Telegram](https://t.me/ali_bek_003)")
st.sidebar.markdown("[Instagram](https://www.instagram.com/alib_ek0311/profilecard/?igsh=MWo5azN2MmM2cGs0aw==)")
st.sidebar.markdown("[Github](https://github.com/AlibekSerikbayev)")
st.write("Ushbu dastur Alibek Serikbayev tomonidan yaratildi")
