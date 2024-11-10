import os
import streamlit as st
import pathlib
from fastai.vision.all import *  

# Windows yo'l muammosini hal qilish
if os.name == 'nt':  # Agar OS Windows bo'lsa
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# Ilovani ishga tushirish uchun ko'rsatmalar
st.set_page_config(page_title="Rasmlar tanish dasturi")
st.title("Rasmlarni tanish dasturi")
st.write("Klasslar avtomobil samolyoti qayiq Yirtqich hayvonlar musiqa asbobi Sport jihozlari Telefon ofis jihozlari oshxona anjomlari")

# Modelni yuklash (Load the model)
@st.cache_data
def load_model():
    model_path = "modelalibek.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please make sure it's in the correct directory.")
    return load_learner(model_path)

try:
    learner = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Rasm yuklash (Upload image)
uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png", "jfif", "webp"])

if uploaded_file is not None:
    # Yuklangan rasmni o'qish (Read uploaded image)
    img = PILImage.create(uploaded_file)
    # Rasmni aniqlash (Predict image)
    try:
        pred, pred_idx, probs = learner.predict(img)
        
        # Natijani ko'rsatish (Show results)
        st.image(img, caption='Yuklangan rasm', use_container_width=True)
        st.write(f"Bu rasm: {pred} (Ishonch: {probs[pred_idx]:.2f})")
    except Exception as e:
        st.error(f"Rasmni aniqlashda xato: {e}")

# Ijtimoiy tarmoq va GitHub sahifalarini ko'rsatish (Display social media and GitHub links)
st.sidebar.header("Qo'shimcha ma'lumotlar")
st.sidebar.write("Bizni ijtimoiy tarmoqlarda kuzatib boring:")
st.sidebar.markdown("[Telegram](https://t.me/ali_bek_003)")
st.sidebar.markdown("[Instagram](https://www.instagram.com/alib_ek0311/profilecard/?igsh=MWo5azN2MmM2cGs0aw==)")
st.sidebar.markdown("[Github](https://github.com/AlibekSerikbayev)")
st.write("Ushbu dastur Alibek Serikbayev tomonidan yaratildi")