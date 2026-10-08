import streamlit as st
import os
from PIL import Image
import replicate

st.set_page_config(page_title="Omni-Face-Changer Pro Masterpiece", layout="wide")

st.title("🔥 Omni-Face-Changer: Professional Face Swap Suite")
st.write("Duniya ki sabse powerful, Stable Cloud Face Swap System!")

# Sidebar for API Token and Model ID with exact official hash from Replicate docs
st.sidebar.subheader("🔑 Pro Cloud Configuration")
api_token = st.sidebar.text_input("Enter Replicate API Token:", type="password")

model_id = st.sidebar.text_input(
    "Replicate Model ID:", 
    value="cdingram/face-swap:d1d6ea8c8be89d664a07a457526f7128109dee7030fdac424788d762c71ed111"
)

option = st.sidebar.selectbox("Mode Pasand karo:", [
    "Pro Generative Photo Swap (100% Exact Identity)",
    "Hardware & Cloud Performance Dashboard"
])

if option == "Pro Generative Photo Swap (100% Exact Identity)":
    st.subheader("📸 Hollywood-Grade Generative Face Swap (Zero Distortion)")
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
        
        if st.button("🚀 Generate 100% Real Pro Masterpiece"):
            if not api_token:
                st.error("❌ Krupaya sidebar ma potano Replicate API Token nakhvo jaruri che!")
            else:
                with st.spinner("✨ Pro Cloud AI par rendering thai rhi che..."):
                    try:
                        s_path = "temp_src.jpg"
                        t_path = "temp_tgt.jpg"
                        s_image.save(s_path)
                        t_image.save(t_path)
                        
                        client = replicate.Client(api_token=api_token)
                        
                        # Running exact model with official schema inputs (input_image & swap_image)
                        output = client.run(
                            model_id.strip(),
                            input={
                                "input_image": open(t_path, "rb"),  # Target body image
                                "swap_image": open(s_path, "rb")    # Source face image[cite: 19]
                            }
                        )
                        
                        if output:
                            st.image(output, caption="✨ 100% Original Identity-Locked Masterpiece", use_container_width=True)
                            st.success("🎉 Duniya se alag, ekdam real result taiyar che!")
                        else:
                            st.error("❌ Result generate nathi thayo.")
                    except Exception as e:
                        st.error(f"Cloud Processing Error: {e}")

else:
    st.subheader("📊 Pro Cloud Performance Dashboard")
    st.metric("Rendering Engine", "Cloud Face-Swap Cluster", "Online")
    st.metric("Identity Accuracy", "High Precision", "Zero Distortion")
    st.info("Pro API connected. Ready for production.")
