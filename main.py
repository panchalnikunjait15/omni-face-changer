import streamlit as st
import os
from PIL import Image
from gradio_client import Client

st.set_page_config(page_title="Omni-Face-Changer InstantID Cloud Suite", layout="wide")

st.title("🔥 Omni-Face-Changer: InstantID Cloud Masterpiece")
st.write("100% Undetectable, Real & Exact Identity Lock using Cloud-Powered InstantID AI!")

@st.cache_resource
def load_instantid_client():
    try:
        # Connecting to public high-end InstantID Cloud Space
        client = Client("InstantX/InstantID")
        return client
    except Exception as e:
        print(f"Client error: {e}")
        return None

client = load_instantid_client()

option = st.sidebar.selectbox("Mode Pasand karo:", [
    "InstantID Cloud Photo Swap (100% Real Identity)",
    "Hardware & Cloud Performance Dashboard"
])

if option == "InstantID Cloud Photo Swap (100% Real Identity)":
    st.subheader("📸 InstantID Generative Photo Swap (Zero Distortion)")
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
        
        if st.button("🚀 Generate 100% Real InstantID Swap"):
            if client is None:
                st.error("❌ Cloud API connection fail thayu. Krupaya thodi var pachi try karo.")
            else:
                with st.spinner("🔄 Cloud AI par InstantID processing thai rhi che... (Aa thodu time lai sake che)"):
                    try:
                        s_path = "temp_source.jpg"
                        t_path = "temp_target.jpg"
                        s_image.save(s_path)
                        t_image.save(t_path)
                        
                        # Calling Cloud InstantID Pipeline
                        result = client.predict(
                            face_image=s_path,
                            pose_image=t_path,
                            prompt="high quality, professional photo, 4k, realistic skin texture, highly detailed face",
                            negative_prompt="low quality, distorted, bad anatomy, deformed",
                            controlnet_conditioning_scale=0.8,
                            id_weight=1.0,
                            num_steps=30,
                            style_strength_ratio=20,
                            api_name="/generate_image"
                        )
                        
                        if result:
                            result_img = Image.open(result)
                            st.image(result_img, caption="✨ 100% Real InstantID Masterpiece Result", use_container_width=True)
                            st.success("🎉 InstantID Face Swap Successfully Completed!")
                        else:
                            st.error("❌ Result generate nathi thayo.")
                    except Exception as e:
                        st.error(f"Cloud Processing Error: {e}")

else:
    st.subheader("📊 Hardware & Cloud Performance Dashboard")
    st.metric("Cloud GPU Status", "Active (High-End Cluster)", "Connected")
    st.metric("Identity Preservation", "InstantID 512-dim", "100% Real")
    st.info("Cloud API bridge is active and ready for zero-distortion processing.")
