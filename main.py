import streamlit as st
from PIL import Image
from gradio_client import Client
import os

st.set_page_config(page_title="Omni-Face-Changer Free Pro Masterpiece", layout="wide")

st.title("🔥 Omni-Face-Changer: 100% Free Hollywood-Grade Face Swap")
st.write("Hugging Face na free AI cluster thi 100% Original Identity-Lock system!")

# Sidebar for Free HF Token
st.sidebar.subheader("🔑 Free Hugging Face Configuration")
hf_token = st.sidebar.text_input("Enter Free HF Token:", type="password")
st.sidebar.caption("Hugging Face parthi free token banavi ne ahi nakhvo (Gated spaces access mate jaruri che).")

option = st.sidebar.selectbox("Mode Pasand karo:", [
    "Pro Generative Photo Swap (100% Free & Original Identity)",
    "System Architecture Dashboard"
])

if option == "Pro Generative Photo Swap (100% Free & Original Identity)":
    st.subheader("📸 Free Hollywood-Grade Generative Face Swap (Zero Cost)")
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
        
        if st.button("🚀 Generate 100% Free Pro Masterpiece"):
            if not hf_token:
                st.warning("⚠️ Krupaya sidebar ma potano free Hugging Face Token nakhvo!")
            else:
                with st.spinner("✨ Free Hugging Face AI Space par rendering thai rhi che... (10-20 sekund lagse)"):
                    try:
                        # Setting environment token for gradio_client authentication
                        os.environ["HF_TOKEN"] = hf_token
                        
                        s_path = "temp_src.jpg"
                        t_path = "temp_tgt.jpg"
                        s_image.save(s_path)
                        t_image.save(t_path)
                        
                        # Connecting to free InstantID Space with token auth
                        client = Client("gokaygokay/InstantID")
                        
                        result = client.predict(
                            input_image=t_path,
                            image_fah=s_path,
                            prompt="high quality, professional portrait, ultra realistic skin pores, 8k resolution, cinematic lighting",
                            negative_prompt="low quality, distorted, bad anatomy, deformed",
                            controlnet_conditioning_scale=0.8,
                            ip_adapter_scale=0.8,
                            num_steps=30,
                            api_name="/generate_image"
                        )
                        
                        if result:
                            st.image(result, caption="✨ 100% Free & Original Identity-Locked Masterpiece", use_container_width=True)
                            st.success("🎉 Ekdam free ane zabardast Hollywood-grade result taiyar che!")
                        else:
                            st.error("❌ Result generate nathi thayo.")
                    except Exception as e:
                        st.error(f"Free Cloud Connection Error: {e}")

else:
    st.subheader("📊 Free Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Hugging Face Free Spaces", "Active")
    st.metric("Billing Status", "100% Free (Zero Cost)", "Active")
    st.info("Running on community AI power with token auth.")
