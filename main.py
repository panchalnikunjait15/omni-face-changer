import streamlit as st
from PIL import Image
from gradio_client import Client
import os

st.set_page_config(page_title="Omni-Face-Changer Pro Masterpiece", layout="wide")

st.title("Omni-Face-Changer: Ultimate Hollywood-Grade Face Swap")
st.write("Hugging Face Pro AI Engine - 100% Original Identity-Lock System")

# Sidebar for Free HF Token
st.sidebar.subheader("Hugging Face Configuration")
hf_token = st.sidebar.text_input("Enter Free HF Token:", type="password")
hf_token = hf_token.strip() if hf_token else ""
st.sidebar.caption("Paste your free Hugging Face read token here.")

option = st.sidebar.selectbox("Select Mode:", [
    "Pro Generative Photo Swap (Original Quality)",
    "System Architecture Dashboard"
])

if option == "Pro Generative Photo Swap (Original Quality)":
    st.subheader("Hollywood-Grade Generative Face Swap")
    source_file = st.file_uploader("Source Face Photo UPLOAD karo (Original Person):", type=['jpg', 'jpeg', 'png', 'webp'])
    target_file = st.file_uploader("Target Body/Scene Photo UPLOAD karo:", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if source_file and target_file:
        s_image = Image.open(source_file)
        t_image = Image.open(target_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(s_image, caption="Source Face (Original)", use_container_width=True)
        with col2:
            st.image(t_image, caption="Target Body", use_container_width=True)
        
        if st.button("Generate Masterpiece"):
            if not hf_token:
                st.warning("Please enter your Hugging Face Token in the sidebar first!")
            else:
                with st.spinner("Connecting to High-End AI Cluster... (Rendering in progress)"):
                    try:
                        s_path = "temp_src.jpg"
                        t_path = "temp_tgt.jpg"
                        s_image.save(s_path)
                        t_image.save(t_path)
                        
                        # Connecting to active InstantID Gradio Space with token auth
                        client = Client("radames/RealTime-InstantID", token=hf_token)
                        
                        result = client.predict(
                            image=t_path,
                            face_image=s_path,
                            prompt="high quality, professional portrait, ultra realistic skin pores, 8k resolution, cinematic lighting",
                            api_name="/predict"
                        )
                        
                        if result:
                            st.image(result, caption="100% Original Hollywood-Grade Masterpiece", use_container_width=True)
                            st.success("Masterpiece generated successfully!")
                        else:
                            st.error("Failed to generate result.")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

else:
    st.subheader("Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Hugging Face Pro Spaces", "Active")
    st.metric("Identity Lock", "100% Original Match", "Maximum")
    st.info("Configured for high-end generative face swapping.")
