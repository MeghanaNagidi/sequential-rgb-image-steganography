import streamlit as st
import os
import torch
from imageio import imread, imwrite
from PIL import Image
import numpy as np

from steganogan.models import SteganoGAN

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="SteganoGAN | Sequential RGB", page_icon="🔐", layout="centered")

# Inject Custom CSS for premium glassmorphism and gradient aesthetics
st.markdown("""
<style>
    /* Main Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0d0e15 0%, #1a1b26 100%);
        color: #e0e2eb;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #7aa2f7;
        font-weight: 700 !important;
    }
    
    /* Custom Header Gradient */
    .title-gradient {
        background: -webkit-linear-gradient(45deg, #7aa2f7, #bb9af7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .subtitle {
        text-align: center;
        color: #a9b1d6;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Containers */
    .glass-container {
        background: rgba(30, 32, 48, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .glass-container:hover {
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7aa2f7 0%, #bb9af7 100%);
        color: #1a1b26 !important;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(122, 162, 247, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='title-gradient'>SteganoGAN</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Sequential RGB Deep Learning Steganography</div>", unsafe_allow_html=True)

# --- MODEL LOADING ---
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        # Force a fresh load by removing streamlit caching
        model = SteganoGAN.load(path=model_path, cuda=False)
        return model
    except Exception as e:
        import traceback
        st.error(f"Critical System Error: {e}")
        st.code(traceback.format_exc())
        return None

# Sidebar for Model Selection Configuration
st.sidebar.markdown("### ⚙️ System Configuration")
default_model_path = os.path.join("models", "sequential_rgb_FINAL.steg")
model_path = st.sidebar.text_input("Checkpoint Path (.steg)", value=default_model_path)

model = load_model(model_path)

if model is None:
    st.error(f"Model not found at `{model_path}`! Please train the model first or provide a valid path.")
    st.stop()
else:
    st.sidebar.success("✅ Model Successfully Loaded!")
    st.sidebar.info("Architecture: Sequential RGB Channel Generation")

# --- UI TABS ---
tab1, tab2 = st.tabs(["🔒 Hide Secret (Encode)", "🔓 Extract Secret (Decode)"])

# --- ENCODE TAB ---
with tab1:
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown("### 1. Upload Cover Image")
    cover_file = st.file_uploader("Choose a high-resolution image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="encode_upload")
    
    st.markdown("### 2. Enter Secret Message")
    secret_message = st.text_area("Type the secret payload you want to embed into the RGB channels:", 
                                  placeholder="E.g., Final Year Project Launch Codes...")
    
    if cover_file is not None and secret_message:
        # Display original image
        st.image(cover_file, caption="Original Cover Image", use_container_width=True)
        
        if st.button("✨ Generate Steganographic Image"):
            with st.spinner("Executing Sequential RGB GAN Matrix... Neural network embedding in progress..."):
                try:
                    # Save uploaded file temporarily
                    temp_cover_path = "temp_cover.png"
                    temp_output_path = "stego_output.png"
                    
                    with open(temp_cover_path, "wb") as f:
                        f.write(cover_file.getbuffer())
                        
                    # Execute Encoding Model
                    model.encode(temp_cover_path, temp_output_path, secret_message)
                    
                    st.success("Message successfully encrypted and embedded stealthily!")
                    
                    # Display resulting image
                    stego_img = Image.open(temp_output_path)
                    st.image(stego_img, caption="Encrypted Stego Image (Mathematically Altered)", use_container_width=True)
                    
                    # Provide Download Button
                    with open(temp_output_path, "rb") as file:
                        btn = st.download_button(
                            label="📥 Download Secure Stego Image",
                            data=file,
                            file_name="secret_stego_image.png",
                            mime="image/png"
                        )
                        
                    # Cleanup
                    os.remove(temp_cover_path)
                    
                except Exception as e:
                    st.error(f"Critical Encoding Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- DECODE TAB ---
with tab2:
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown("### 1. Upload Stego Image")
    stego_file = st.file_uploader("Upload an encoded image generated by this network", type=["png"], key="decode_upload")
    
    if stego_file is not None:
        st.image(stego_file, caption="Target Image for Analysis", use_container_width=True)
        
        if st.button("🕵️ Extract Hidden Data"):
            with st.spinner("Decoding sequential RGB channels... Extracting payload..."):
                try:
                    temp_stego_path = "temp_decode.png"
                    with open(temp_stego_path, "wb") as f:
                        f.write(stego_file.getbuffer())
                        
                    # Execute Decoding Model
                    extracted_text = model.decode(temp_stego_path)
                    
                    st.balloons()
                    st.success("Secret Payload Successfully Reconstructed!")
                    st.info(f"**Extracted Message:**\n\n{extracted_text}")
                    
                    # Cleanup
                    os.remove(temp_stego_path)
                    
                except ValueError as ve:
                    st.error("Extraction Failed: Could not locate a coherent message. Either the image is not encoded, heavily modified, or the model isn't fully trained.")
                except Exception as e:
                    st.error(f"Critical Decoding Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
