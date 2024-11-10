#venv\Scripts\activate
#streamlit run ./app.py
#bu o'zgarish
import streamlit as st
import pathlib
import plotly.express as px
from fastai.vision.all import *  # Import fastai

# Fix for Windows path issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Add instructions for running the app
st.set_page_config(page_title="Rasmlar tanish dasturi")
st.title("Rasmlarni tanish dasturi")
st.write("Klasslar Car Airplane Boat Carnivore Musical_instrument Sports_equipment Telephone Office_supplies Kitchen_utensil")

# Modelni yuklash (Load the model)
@st.cache_resource
def load_model():
    return load_learner("modelalibek.pkl")

try:
    learner = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please make sure it's in the same directory as this script.")
    st.stop()

# Rasm yuklash (Upload image)
uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png","jfif","webp"])

if uploaded_file is not None:
    # Yuklangan rasmni o'qish (Read uploaded image)
    img = PILImage.create(uploaded_file)
    # Rasmni aniqlash (Predict image)
    try:
        pred, pred_idx, probs = learner.predict(img)
        
        # Natijani ko'rsatish (Show results)
        st.image(img, caption='Yuklangan rasm', use_container_width=True)  # Updated parameter
        st.write(f"Bu rasm: {pred} (Ishonch: {probs[pred_idx]:.2f})")
    except Exception as e:
        st.error(f"Rasmni aniqlashda xato: {e}")

# Ijtimoiy tarmoq va GitHub sahifalarini ko'rsatish (Display social media and GitHub links)
st.sidebar.header("Qo'shimcha ma'lumotlar")
st.sidebar.write("Bizni ijtimoiy tarmoqlarda kuzatib boring:")
st.sidebar.markdown("[Telegram](https://t.me/ali_bek_003)")
st.sidebar.markdown("[Instagram](https://www.instagram.com/alib_ek0311/profilecard/?igsh=MWo5azN2MmM2cGs0aw==)")
st.sidebar.markdown("[Github](https://github.com/AlibekSerikbayev)")
st.write("Ushbu dastur Alibek Serikbayev tomonidan yaratildi ")