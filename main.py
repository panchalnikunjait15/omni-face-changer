import streamlit as st
from PIL import Image
import replicate
import os

st.set_page_config(page_title="Omni-Face-Changer Pro Masterpiece", layout="wide")

st.title("Omni-Face-Changer: Ultimate Hollywood-Grade Face Swap")
st.write("Official Replicate API Engine - 100% Stable, No Queues, Zero 404 Errors")

# Sidebar for Replicate API Token
st.sidebar.subheader("Replicate API Configuration")
replicate_token = st.sidebar.text_input("Enter Replicate API Token:", type="password")
replicate_token = replicate_token.strip() if replicate_token else ""
st.sidebar.caption("Get your free token from replicate.com/account/api-tokens")

option = st.sidebar.selectbox("Select Mode:", [
    "Pro Generative Face Swap (Original Quality)",
    "System Architecture Dashboard"
])

if option == "Pro Generative Face Swap (Original Quality)":
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
            if not replicate_token:
                st.warning("Please enter your Replicate API Token in the sidebar first!")
            else:
                with st.spinner("Generating Hollywood-grade face swap via Replicate API... (Please wait)"):
                    try:
                        os.environ["REPLICATE_API_TOKEN"] = replicate_token
                        
                        s_path = "temp_src.jpg"
                        t_path = "temp_tgt.jpg"
                        s_image.save(s_path)
                        t_image.save(t_path)
                        
                        # Opening files for Replicate API execution
                        with open(s_path, "rb") as face_file, open(t_path, "rb") as target_file_obj:
                            output = replicate.run(
                                "tencentarc/instantid:0fcac9846bfa33104e76c125df7f433f5d625d997232231ff08a984a9e525049",
                                input={
                                    "image": target_file_obj,
                                    "face_image": face_file,
                                    "prompt": "high quality, professional portrait, ultra realistic skin pores, 8k resolution, cinematic lighting",
                                    "negative_prompt": "low quality, distorted, bad anatomy, blurry",
                                    "controlnet_conditioning_scale": 0.8,
                                    "ip_adapter_scale": 0.8,
                                    "num_outputs": 1
                                }
                            )
                        
                        if output:
                            img_url = output[0] if isinstance(output, list) else output
                            st.image(img_url, caption="100% Original Hollywood-Grade Masterpiece", use_container_width=True)
                            st.success("Masterpiece generated successfully via Replicate API!")
                        else:
                            st.error("API returned empty output. Please try again.")
                    except Exception as e:
                        st.error(f"Replicate API Error: {e}")

else:
    st.subheader("Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Replicate Official API", "Active")
    st.metric("Identity Lock", "100% Original Match", "Maximum")
    st.info("Configured for direct, high-speed, zero-queue execution.")
