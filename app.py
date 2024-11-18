import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
from pathlib import Path

# Pathlib uchun moslik o‘rnatish
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath if pathlib.Path().drive else temp

# Ilova sarlavhasi
st.title('Transportni klassifikatsiya qiluvchi model')

# Fayl yuklash
files = st.file_uploader("Rasm yuklash", type=["png", "jpeg", "jpg"])

if files:
    # Yuklangan rasmni chiqarish
    st.image(files, caption="Yuklangan rasm", use_column_width=True)

    # Faylni PILImage formatiga aylantirish
    img = PILImage.create(files.getvalue())

    # Modelni yuklash
    try:
        model = load_learner('transport_model.pkl')
        
        # Bashorat qilish
        pred, pred_id, probs = model.predict(img)
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")
        
        # Diagramma
        fig = px.bar(x=probs * 100, y=model.dls.vocab, orientation='h', title="Bashorat ehtimolligi")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Modelni yuklashda xatolik: {e}")


# import streamlit as st
# from fastai.vision.all import *
# import plotly.express as px
# import pathlib
# pathlib.PosixPath = pathlib.Path

# # title
# st.title('Transportni klassifikatsiya qiluvchi model')

# # Rasmni joylash
# files = st.file_uploader("Rasm yuklash", type=["avif", "png", "jpeg", "gif", "svg"])
# if files:
#     st.image(files)  # rasmni chiqarish
#     # PIL convert
#     img = PILImage.create(files)
    
#     # Modelni yuklash
#     model = load_learner('transport_model.pkl')

#     # Bashorat qiymatni topamiz
#     pred, pred_id, probs = model.predict(img)
#     st.success(f"Bashorat: {pred}")
#     st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")

#     # Plotting
#     fig = px.bar(x=probs * 100, y=model.dls.vocab)
#     st.plotly_chart(fig)