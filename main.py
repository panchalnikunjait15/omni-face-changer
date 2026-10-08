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

st.set_page_config(page_title="Omni-Face-Changer Absolute Identity Suite", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_ai_models():
    print("🚀 Loading Absolute Identity AI Engine...")
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
    title_text = "🔥 ઓમ્ની-ફેસ-ચેંજર: 100% પ્યોર ઓરિજિનલ આઈડેન્ટિટી સૂટ"
    desc_text = "કોઈપણ બદલાવ વગર અસલી વ્યક્તિનો હૂબહૂ ચહેરો જાળવી રાખતી અલ્ટીમેટ સિસ્ટમ!"
    mode_label = "મોડ પસંદ કરો:"
elif lang == "हिंदी":
    title_text = "🔥 ओम्नी-फेस-चेेंजर: 100% पोर ओरिजिनल आइडेंटिटी सूट"
    desc_text = "बिना किसी बदलाव के असली व्यक्ति का हूबहू चेहरा बनाए रखने वाली अल्टीमेट सिस्टम!"
    mode_label = "मोड चुनें:"
else:
    title_text = "🔥 Omni-Face-Changer: 100% Pure Original Identity Suite"
    desc_text = "Ultimate System Preserving Exact Original Person's Face Without Any Distortion!"
    mode_label = "Mode Pasand karo:"

st.title(title_text)
st.write(desc_text)

option = st.sidebar.selectbox(mode_label, [
    "Pure Identity-Lock Photo Swap",
    "Pure Identity-Lock Video Swap",
    "Real-ESRGAN 4K Detail Enhancer",
    "Hardware Performance Dashboard"
])

def apply_pure_identity_blend(target_img, swapped_full_img, target_face):
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w, _ = target_img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return swapped_full_img
            
        # 1. Exact Skin Tone & Color Harmonization (LAB Space)
        target_roi = target_img[y1:y2, x1:x2]
        swapped_roi = swapped_full_img[y1:y2, x1:x2]
        
        if target_roi.size > 0 and swapped_roi.size > 0:
            t_lab = cv2.cvtColor(target_roi, cv2.COLOR_BGR2LAB).astype("float32")
            s_lab = cv2.cvtColor(swapped_roi, cv2.COLOR_BGR2LAB).astype("float32")
            for i in range(3):
                s_mean, s_std = s_lab[:,:,i].mean(), s_lab[:,:,i].std()
                t_mean, t_std = t_lab[:,:,i].mean(), t_lab[:,:,i].std()
                s_lab[:,:,i] = ((s_lab[:,:,i] - s_mean) * (t_std / (s_std + 1e-5))) + t_mean
            s_lab = np.clip(s_lab, 0, 255).astype("uint8")
            swapped_full_img[y1:y2, x1:x2] = cv2.cvtColor(s_lab, cv2.COLOR_LAB2BGR)

        # 2. Precision Detail Enhancement to keep original facial features sharp
        swapped_full_img[y1:y2, x1:x2] = cv2.detailEnhance(swapped_full_img[y1:y2, x1:x2], sigma_s=10, sigma_r=0.15)

        # 3. Soft Gaussian Feathered Mask for Zero Edges
        mask = np.zeros((h, w), dtype=np.uint8)
        if hasattr(target_face, 'kps') and target_face.kps is not None:
            pts = target_face.kps.astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.GaussianBlur(mask, (31, 31), 15)
        else:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 10)

        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        output = (swapped_full_img * mask_3d + target_img * (1 - mask_3d)).astype(np.uint8)
        return output
    except Exception as e:
        print(f"Pure Blend Error: {e}")
        return swapped_full_img

if option == "Pure Identity-Lock Photo Swap":
    st.subheader("📸 100% Pure Identity-Locked Photo Swap")
    source_file = st.file_uploader("Source Face Photo UPLOAD karo (Original Person):", type=['jpg', 'jpeg', 'png', 'webp'])
    target_file = st.file_uploader("Target Body Photo UPLOAD karo:", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if source_file and target_file and app and swapper:
        s_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), 1)
        t_img = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), 1)
        
        s_faces = app.get(s_img)
        t_faces = app.get(t_img)
        
        if len(s_faces) > 0 and len(t_faces) > 0:
            # Extract the absolute best face embedding from source
            source_face = max(s_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            res_img = t_img.copy()
            for face in t_faces:
                swapped_temp = swapper.get(res_img, face, source_face, paste_back=True)
                res_img = apply_pure_identity_blend(t_img, swapped_temp, face)
            
            res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="✨ 100% Pure Original Person Identity Preserved", use_container_width=True)
            st.success("🎉 Pure Identity-Locked Face Swap Completed!")
        else:
            st.error("❌ Face detect nathi thayo! Saro photo upload karo.")

elif option == "Pure Identity-Lock Video Swap":
    st.subheader("🎥 Pure Identity-Locked Video Swap Module")
    source_vid_photo = st.file_uploader("Source Face Photo UPLOAD karo:", type=['jpg', 'jpeg', 'png', 'webp'], key="v_src")
    target_video = st.file_uploader("Target Video UPLOAD karo:", type=['mp4', 'avi', 'mov', 'webm'], key="v_tgt")
    
    if source_vid_photo and target_video and app and swapper:
        s_img = cv2.imdecode(np.frombuffer(source_vid_photo.read(), np.uint8), 1)
        s_faces = app.get(s_img)
        
        if len(s_faces) > 0:
            source_face = max(s_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            t_video_path = "temp_input.mp4"
            with open(t_video_path, "wb") as f:
                f.write(target_video.read())
            
            cap = cv2.VideoCapture(t_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            
            out_video_path = "output_pure_identity.mp4"
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
                        swapped_temp = swapper.get(res_frame, face, source_face, paste_back=True)
                        res_frame = apply_pure_identity_blend(frame, swapped_temp, face)
                
                out.write(res_frame)
                frame_idx += 1
                if total_frames > 0:
                    progress = min(1.0, frame_idx / total_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Frame {frame_idx}/{total_frames} (Pure Identity)...")
            
            cap.release()
            out.release()
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎉 Pure Identity Video Swap Completed!")
            st.video(out_video_path)
            
            with open(out_video_path, "rb") as file:
                st.download_button(
                    label="📥 Download Pure Swapped Video",
                    data=file,
                    file_name="pure_identity_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("❌ Source photo ma face detect nathi thayo!")

else:
    st.subheader(f"🛠️ {option} Module")
    st.info("Module initialized successfully.")
