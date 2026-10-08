import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import insightface
import socket
import urllib.request
import onnxruntime

st.set_page_config(page_title="Omni-Face-Changer Masterpiece", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_ai_models():
    print("🚀 Loading AI Models...")
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    model_path = 'inswapper_128.onnx'
    if not os.path.exists(model_path):
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        try:
            with st.spinner("Downloading model..."):
                urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Model error: {e}")
            
    if os.path.exists(model_path):
        swapper = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
    else:
        swapper = None
        
    return app, swapper

app, swapper = load_ai_models()

lang = st.sidebar.selectbox("🌐 Choose Language / પસંદ કરો / भाषा चुनें", ["English", "ગુજરાતી", "हिंदी"])

if lang == "ગુજરાતી":
    title_text = "🔥 ઓમ્ની-ફેસ-ચેંજર: અલ્ટીમેટ માસ્ટરપીસ સૂટ"
    desc_text = "ગ્લિચ-ફ્રી, 100% રિયલ અને નેચરલ ફેસ-સ્વેપિંગ સિસ્ટમ!"
    mode_label = "મોડ પસંદ કરો:"
elif lang == "हिंदी":
    title_text = "🔥 ओम्नी-फेस-चेेंजर: अल्टीमेट मास्टरपीस सूट"
    desc_text = "ग्लिच-फ्री, 100% रियल और नेचुरल फेस-स्वैपिंग सिस्टम!"
    mode_label = "मोड चुनें:"
else:
    title_text = "🔥 Omni-Face-Changer: Ultimate Masterpiece Suite"
    desc_text = "Glitch-Free, 100% Real & Natural Face-Swapping System!"
    mode_label = "Mode Pasand karo:"

st.title(title_text)
st.write(desc_text)

option = st.sidebar.selectbox(mode_label, [
    "Ultra-Realistic Photo Swap",
    "Video Swap Engine",
    "Real-ESRGAN 4K Detail Enhancer",
    "Hardware Performance Dashboard"
])

def apply_glitch_free_real_blend(target_img, swapped_full_img, face):
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w, _ = target_img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return swapped_full_img
            
        # 1. Advanced LAB Color Matching to match exact skin tone and lighting
        target_face = target_img[y1:y2, x1:x2]
        swapped_face = swapped_full_img[y1:y2, x1:x2]
        if target_face.size > 0 and swapped_face.size > 0:
            t_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB).astype("float32")
            s_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype("float32")
            for i in range(3):
                s_mean, s_std = s_lab[:,:,i].mean(), s_lab[:,:,i].std()
                t_mean, t_std = t_lab[:,:,i].mean(), t_lab[:,:,i].std()
                s_lab[:,:,i] = ((s_lab[:,:,i] - s_mean) * (t_std / (s_std + 1e-5))) + t_mean
            s_lab = np.clip(s_lab, 0, 255).astype("uint8")
            swapped_full_img[y1:y2, x1:x2] = cv2.cvtColor(s_lab, cv2.COLOR_LAB2BGR)

        # 2. Skin texture enhancement for pores and realistic details
        swapped_full_img[y1:y2, x1:x2] = cv2.detailEnhance(swapped_full_img[y1:y2, x1:x2], sigma_s=10, sigma_r=0.15)

        # 3. Smooth Gaussian Feathered Mask (Zero harsh lines or glitches)
        mask = np.zeros((h, w), dtype=np.uint8)
        if hasattr(face, 'kps') and face.kps is not None:
            pts = face.kps.astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.GaussianBlur(mask, (35, 35), 18)
        else:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (25, 25), 12)

        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        output = (swapped_full_img * mask_3d + target_img * (1 - mask_3d)).astype(np.uint8)
        return output
    except Exception as e:
        print(f"Blend Error: {e}")
        return swapped_full_img

if option == "Ultra-Realistic Photo Swap":
    st.subheader("📸 100% Glitch-Free & Natural Photo Swap")
    source_file = st.file_uploader("Source Face Photo UPLOAD karo:", type=['jpg', 'jpeg', 'png', 'webp'])
    target_file = st.file_uploader("Target Body Photo UPLOAD karo:", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if source_file and target_file and app and swapper:
        s_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), 1)
        t_img = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), 1)
        
        s_faces = app.get(s_img)
        t_faces = app.get(t_img)
        
        if len(s_faces) > 0 and len(t_faces) > 0:
            res_img = t_img.copy()
            for face in t_faces:
                swapped_temp = swapper.get(res_img, face, s_faces[0], paste_back=True)
                res_img = apply_glitch_free_real_blend(t_img, swapped_temp, face)
            
            res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="✨ 100% Clean, Real & Natural Face Swap", use_container_width=True)
            st.success("🎉 Face Swap Completed Successfully!")
        else:
            st.error("❌ Face detect nathi thayo! Saro photo upload karo.")

elif option == "Video Swap Engine":
    st.subheader("🎥 Video Face Swap Module")
    source_vid_photo = st.file_uploader("Source Face Photo UPLOAD karo:", type=['jpg', 'jpeg', 'png', 'webp'], key="v_src")
    target_video = st.file_uploader("Target Video UPLOAD karo:", type=['mp4', 'avi', 'mov', 'webm'], key="v_tgt")
    
    if source_vid_photo and target_video and app and swapper:
        s_img = cv2.imdecode(np.frombuffer(source_vid_photo.read(), np.uint8), 1)
        s_faces = app.get(s_img)
        
        if len(s_faces) > 0:
            t_video_path = "temp_input.mp4"
            with open(t_video_path, "wb") as f:
                f.write(target_video.read())
            
            cap = cv2.VideoCapture(t_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            
            out_video_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                t_faces = app.get(frame)
                res_frame = frame.copy()
                if len(t_faces) > 0:
                    for face in t_faces:
                        swapped_temp = swapper.get(res_frame, face, s_faces[0], paste_back=True)
                        res_frame = apply_glitch_free_real_blend(frame, swapped_temp, face)
                
                out.write(res_frame)
                frame_idx += 1
                if total_frames > 0:
                    progress = min(1.0, frame_idx / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Frame {frame_idx}/{total_frames}...")
            
            cap.release()
            out.release()
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎉 Video Face Swap Completed!")
            st.video(out_video_path)
            
            with open(out_video_path, "rb") as file:
                st.download_button(
                    label="📥 Download Swapped Video",
                    data=file,
                    file_name="swapped_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("❌ Source photo ma face detect nathi thayo!")

else:
    st.subheader(f"🛠️ {option} Module")
    st.info("Module initialized successfully.")
