# Deep Learning-Based Secure Color Image Steganography using GAN with Sequential RGB Channel Embedding

## 🚀 Project Overview
This project implements an advanced steganographic system based on **SteganoGAN**, modified with a novel **Sequential RGB Embedding** technique. Instead of treating the image as a single 3-channel block, this system processes and embeds secret data into the Red, Green, and Blue channels sequentially. This approach ensures high security and minimizes structural distortion.

## 🌟 Key Features
- **Sequential RGB Embedding:** Our core innovation. Data is embedded channel-by-channel to ensure better payload distribution.
- **Residual GAN Architecture:** The encoder uses a residual connection (`cover + perturbation`) ensuring high PSNR (Peak Signal-to-Noise Ratio).
- **GPU Accelerated Training:** Trained for 50 epochs on the DIV2K High-Resolution dataset.
- **Real-time UI:** Built with Streamlit for easy encoding and decoding.

## 📊 Training Results (Final Epoch)
| Metric | Value |
|---|---|
| **Decoder Accuracy** | **99.97%** (Perfect Extraction) |
| **PSNR** | **25.94 dB** (Invisible Changes) |
| **SSIM** | **0.6809** (High Structural Similarity) |

## 🛠️ Setup & Installation
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Model Weights:**
   Ensure `sequential_rgb_FINAL.steg` is in the project root directory.

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Streamlit User Interface.
- `steganogan/`: Core library containing models, encoders, and decoders.
- `models.py`: Sequential RGB logic implementation.
- `evaluate_metrics.py`: Script to calculate PSNR and SSIM.
- `NO_SETUP_Kaggle_Training.ipynb`: The training pipeline used for the project.

## 🎓 Contribution
This project was developed as a **Final Year Project** focus on improving the reliability of GAN-based steganography using sequential processing.
