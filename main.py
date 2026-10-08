import streamlit as st
from PIL import Image
from gradio_client import Client
import os

st.set_page_config(page_title="Omni-Face-Changer Free Pro Masterpiece", layout="wide")

st.title("🔥 Omni-Face-Changer: 100% Free Hollywood-Grade Face Swap")
st.write("Bina koi paisa kharchya, Hugging Face na free AI cluster thi 100% Original Identity-Lock system!")

# Mode Selection in Sidebar
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
            with st.spinner("✨ Free Hugging Face AI Space par rendering thai rhi che... (Ahi 10-20 sekund lagi shake, kem ke aa ekdam free public cluster che)"):
                try:
                    s_path = "temp_src.jpg"
                    t_path = "temp_tgt.jpg"
                    s_image.save(s_path)
                    t_image.save(t_path)
                    
                    # Connecting to a public free InstantID Space via gradio_client (Zero Cost!)
                    client = Client("gokaygokay/InstantID")
                    
                    result = client.predict(
                        input_image=t_path,
                        image_fah=s_path,  # Source face image
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
                    # Alternative fallback public space if primary is busy
                    try:
                        client_alt = Client("TencentARC/InstantID")
                        result_alt = client_alt.predict(
                            face_image=s_path,
                            pose_image=t_path,
                            prompt="high quality, professional portrait, ultra realistic skin pores, 8k resolution",
                            negative_prompt="low quality, distorted",
                            scale=1.2,
                            num_steps=30,
                            api_name="/generate"
                        )
                        if result_alt:
                            st.image(result_alt, caption="✨ 100% Free Masterpiece (Alt Server)", use_container_width=True)
                            st.success("🎉 Result aavi gayo!")
                        else:
                            st.error("❌ Alternate server par pan result nathi malyo.")
                    except Exception as e2:
                        st.error(f"Free Cloud Connection Error: {e} | {e2}")

else:
    st.subheader("📊 Free Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Hugging Face Free Spaces", "Active")
    st.metric("Billing Status", "100% Free (Zero Cost)", "Active")
    st.info("No API token required. Running on open-source community AI power.")
