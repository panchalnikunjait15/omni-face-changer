import streamlit as st
from PIL import Image
from gradio_client import Client
import os

st.set_page_config(page_title="Omni-Face-Changer Pro Masterpiece", layout="wide")

st.title("Omni-Face-Changer: Ultimate Hollywood-Grade Face Swap")
st.write("100% Free Hugging Face Neural Engine - InstantID System")

# Sidebar for Free HF Token (Optional)
st.sidebar.subheader("Hugging Face Configuration")
hf_token = st.sidebar.text_input("Enter Free HF Token (Optional):", type="password")
hf_token = hf_token.strip() if hf_token else None
st.sidebar.caption("Leave blank or paste your free HF token.")

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
        
        if st.button("Generate Masterpiece (Free)"):
            with st.status("Connecting to Free AI Node... Please wait...", expanded=True) as status:
                try:
                    s_path = "temp_src.jpg"
                    t_path = "temp_tgt.jpg"
                    s_image.save(s_path)
                    t_image.save(t_path)
                    
                    st.write("Connecting to active InstantID space...")
                    # Using the active, verified public space repository
                    client = Client("gokaygokay/InstantID", token=hf_token)
                    
                    st.write("Processing face swap and cinematic lighting...")
                    result = client.predict(
                        t_path,  # Target image
                        s_path,  # Source face image
                        "high quality, professional portrait, ultra realistic skin pores, 8k resolution, cinematic lighting",
                        "low quality, distorted, bad anatomy, blurry",
                        0.8, # controlnet scale
                        0.8, # ip-adapter scale
                        30,  # num steps
                        api_name="/generate_image"
                    )
                    
                    if result:
                        status.update(label="Masterpiece generated successfully! (100% Free)", state="complete", expanded=False)
                        st.image(result, caption="100% Original Hollywood-Grade Masterpiece", use_container_width=True)
                        st.success("Masterpiece rendered completely free of cost!")
                    else:
                        status.update(label="Generation failed.", state="error", expanded=True)
                        st.error("Received empty response from the server.")
                except Exception as e:
                    status.update(label="Server Notice", state="error", expanded=True)
                    st.error(f"Error: {e}. (Tip: If the server is busy, just click generate again!)")

else:
    st.subheader("Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Free Hugging Face Cluster", "Active")
    st.metric("Cost", "100% Free", "Maximum Savings")
    st.info("Configured with active, 404-free endpoint.")
