import streamlit as st
from gradio_client import Client
import os

st.set_page_config(page_title="Omni-Face Studio", layout="centered")

st.title("🎬 Omni-Face Studio")
st.write("Professional Face-Swap Web App")

# Colab .gradio.live link input
colab_url = st.text_input("🔗 Enter Google Colab Live Link (.gradio.live):", value="")

col1, col2 = st.columns(2)
with col1:
    source_file = st.file_uploader("1. Source Face (Your Photo)", type=["jpg", "jpeg", "png"])
with col2:
    target_file = st.file_uploader("2. Target Body (Target Photo)", type=["jpg", "jpeg", "png"])

if source_file and target_file:
    st.image([source_file, target_file], caption=["Source Face", "Target Body"], width=200)
    
    if st.button("🚀 Generate Face Swap", type="primary"):
        if not colab_url:
            st.error("Please enter your Google Colab .gradio.live link first!")
        else:
            with st.spinner("Processing on your private T4 GPU..."):
                try:
                    # Save temporary images
                    with open("source.jpg", "wb") as f:
                        f.write(source_file.getbuffer())
                    with open("target.jpg", "wb") as f:
                        f.write(target_file.getbuffer())
                    
                    # Connect to Google Colab backend
                    client = Client(colab_url)
                    result = client.predict(
                        source_img="source.jpg",
                        target_img="target.jpg",
                        api_name="/swap_face"
                    )
                    
                    st.success("Masterpiece Generated Successfully!")
                    st.image(result, caption="Hollywood Masterpiece", use_container_width=True)
                except Exception as e:
                    st.error(f"Connection or Processing Error: {e}")
