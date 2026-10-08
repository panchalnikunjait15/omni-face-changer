import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import insightface
import socket
import time
import urllib.request
import onnxruntime

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_ai_models():
    print("🚀 AI Models load thai rhi che...")
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
    # 'buffalo_s' vapryu che jethi 512MB RAM limit andar aram thi chale!
    app = FaceAnalysis(name='buffalo_s', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    model_path = 'inswapper_128.onnx'
    if not os.path.exists(model_path):
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        try:
            with st.spinner("AI Model (inswapper_128.onnx) download thai rhi che..."):
                urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Model download error: {e
