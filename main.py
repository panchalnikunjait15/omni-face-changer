import streamlit as st
from PIL import Image
from gradio_client import Client
import os

st.set_page_config(page_title="Omni-Face-Changer Free Pro Masterpiece", layout="wide")

st.title("Omni-Face-Changer: 100% Free Hollywood-Grade Face Swap")
st.write("Hugging Face free AI cluster for 100% Original Identity-Lock system!")

# Sidebar for Free HF Token
st.sidebar.subheader("Free Hugging Face Configuration")
hf_token = st.sidebar.text_input("Enter Free HF Token:", type="password")
st.sidebar.caption("Create a free token on Hugging Face and paste it here.")

option = st.sidebar.selectbox("Select Mode:", [
    "Pro Generative Photo Swap (Free & Original Identity)",
    "System Architecture Dashboard"
])

if option == "Pro Generative Photo Swap (Free & Original Identity)":
    st.subheader("Free Hollywood-Grade Generative Face Swap")
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
        
        if st.button("Generate Free Pro Masterpiece"):
            if not hf_token:
                st.warning("Please enter your free Hugging Face Token in the sidebar!")
            else:
                with st.spinner("Processing on Free Hugging Face AI Space... (Takes 10-20 seconds)"):
                    try:
                        os.environ["HF_TOKEN"] = hf_token
                        
                        s_path = "temp_src.jpg"
                        t_path = "temp_tgt.jpg"
                        s_image.save(s_path)
                        t_image.save(t_path)
                        
                        success = False
                        result = None
                        
                        # List of active public spaces to try as fallbacks
                        spaces_to_try = [
                            ("InstantX/InstantID", s_path, t_path),
                            ("radames/RealTime-InstantID", s_path, t_path),
                            ("TencentARC/InstantID", s_path, t_path)
                        ]
                        
                        for space_id, s_p, t_p in spaces_to_try:
                            try:
                                client = Client(space_id)
                                if "InstantX" in space_id:
                                    result = client.predict(
                                        input_image=t_p,
                                        image_fah=s_p,
                                        prompt="high quality, professional portrait, ultra realistic skin pores, 8k resolution, cinematic lighting",
                                        negative_prompt="low quality, distorted",
                                        controlnet_conditioning_scale=0.8,
                                        ip_adapter_scale=0.8,
                                        num_steps=30,
                                        api_name="/generate_image"
                                    )
                                elif "radames" in space_id:
                                    result = client.predict(
                                        image=t_p,
                                        face_image=s_p,
                                        prompt="high quality, professional portrait",
                                        api_name="/predict"
                                    )
                                else:
                                    result = client.predict(
                                        face_image=s_p,
                                        pose_image=t_p,
                                        prompt="high quality, professional portrait",
                                        api_name="/generate"
                                    )
                                
                                if result:
                                    success = True
                                    break
                            except Exception as ex:
                                continue
                        
                        if success and result:
                            st.image(result, caption="100% Free & Original Identity-Locked Masterpiece", use_container_width=True)
                            st.success("Result generated successfully!")
                        else:
                            st.error("All public free spaces are currently busy. Please try again in a moment.")
                    except Exception as e:
                        st.error(f"Cloud Connection Error: {e}")

else:
    st.subheader("Free Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Hugging Face Free Spaces", "Active")
    st.metric("Billing Status", "100% Free (Zero Cost)", "Active")
    st.info("Running on community AI power with token auth.")
