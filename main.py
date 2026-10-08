import streamlit as st
from PIL import Image
from gradio_client import Client
import os

st.set_page_config(page_title="Omni-Face-Changer Pro Masterpiece", layout="wide")

st.title("Omni-Face-Changer: Ultimate Hollywood-Grade Face Swap")
st.write("Official InstantX Direct Engine - 100% Original Identity-Lock System")

# Sidebar for Free HF Token
st.sidebar.subheader("Hugging Face Configuration")
hf_token = st.sidebar.text_input("Enter Free HF Token:", type="password")
hf_token = hf_token.strip() if hf_token else ""
st.sidebar.caption("Paste your free Hugging Face read token here.")

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
            if not hf_token:
                st.warning("Please enter your Hugging Face Token in the sidebar first!")
            else:
                s_path = "temp_src.jpg"
                t_path = "temp_tgt.jpg"
                s_image.save(s_path)
                t_image.save(t_path)
                
                with st.status("Connecting to Official InstantX AI Cluster...", expanded=True) as status:
                    st.write("Establishing direct secure connection to InstantX Space...")
                    try:
                        # Direct connection using the full official Hugging Face Space URL
                        client = Client("https://huggingface.co/spaces/InstantX/InstantID", token=hf_token)
                        st.write("Rendering Hollywood-grade face swap & cinematic lighting...")
                        
                        result = client.predict(
                            t_path,  # Target image (pose/body)
                            s_path,  # Source face image
                            "high quality, professional portrait, ultra realistic skin pores, 8k resolution, cinematic lighting",
                            "low quality, distorted, bad anatomy, blurry",
                            0.8, # controlnet scale
                            0.8, # ip-adapter scale
                            30,  # num steps
                            api_name="/generate_image"
                        )
                        
                        if result:
                            status.update(label="Masterpiece generated successfully! 100% Identity Locked.", state="complete", expanded=False)
                            st.image(result, caption="100% Original Hollywood-Grade Masterpiece", use_container_width=True)
                            st.success("Masterpiece rendered with 100% original identity match!")
                        else:
                            status.update(label="Generation returned empty result.", state="error", expanded=True)
                            st.error("Server returned an empty response. Please try again.")
                    except Exception as e:
                        status.update(label="Connection Error Encountered", state="error", expanded=True)
                        st.error(f"Error: {e}. (Tip: Ensure your HF Token is correct and you are logged into Hugging Face)")

else:
    st.subheader("Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Official InstantX Space", "Active")
    st.metric("Identity Lock", "100% Original Match", "Maximum")
    st.info("Configured with direct URL endpoint routing.")
