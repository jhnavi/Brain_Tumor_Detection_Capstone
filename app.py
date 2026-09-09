 from pathlib import Path
import json, numpy as np, tensorflow as tf, streamlit as st
from PIL import Image
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"models/brain_tumor_mobilenetv2.keras"
st.set_page_config(page_title="Brain Tumor Detection",page_icon="🧠")
st.title("🧠 Brain Tumor Detection")
st.warning("Educational prototype only — not a medical diagnostic device.")
if not MODEL.exists(): st.error("Train the model first: python src/train_model.py"); st.stop()
model=tf.keras.models.load_model(MODEL)
file=st.file_uploader("Upload MRI image",type=["jpg","jpeg","png"])
if file:
    im=Image.open(file).convert("RGB"); st.image(im,caption="Uploaded MRI",use_container_width=True)
    if st.button("Analyze MRI",use_container_width=True):
        x=np.expand_dims(np.array(im.resize((224,224)),dtype=np.float32),0)
        p=float(model.predict(x,verbose=0)[0][0])
        label="Tumor detected" if p>=.5 else "No tumor detected"
        conf=p*100 if p>=.5 else (1-p)*100
        st.subheader(label); st.metric("Model confidence",f"{conf:.2f}%"); st.progress(int(conf))
