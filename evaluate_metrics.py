import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from imageio import imread
from PIL import Image
from torch.nn.functional import mse_loss

from steganogan.models import SteganoGAN
from steganogan.utils import ssim

def calculate_psnr(mse):
    if mse == 0:
        return float('inf')
    # Because SteganoGAN normalizes images between -1.0 and 1.0, the maximum pixel value difference is 2.
    # Therefore, the MAX^2 for the PSNR formula is 2^2 = 4.
    return 10 * torch.log10(4 / mse).item()

def evaluate(model_path, image_dir, message="Test message for final year project", limit=None):
    print(f"Loading Sequential-RGB model from {model_path}...")
    steganogan = SteganoGAN.load(path=model_path, cuda=False)
    
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the specified directory.")
        return
        
    if limit is not None:
        images = images[:limit]
        
    print(f"Found {len(images)} images to evaluate.")
    
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    success_count = 0
    
    for img_name in tqdm(images, desc="Evaluating Images"):
        img_path = os.path.join(image_dir, img_name)
        try:
            # Prepare the cover image with resizing to 512x512 for memory efficiency
            img = Image.open(img_path).convert('RGB').resize((512, 512))
            cover = np.array(img) / 127.5 - 1.0
            cover = torch.FloatTensor(cover).permute(2, 0, 1).unsqueeze(0)
            
            # Use resized dimensions
            _, _, h, w = cover.size()
            
            # Format the sequential RGB payload
            payload = steganogan._make_payload(w, h, steganogan.data_depth * 3, message)
            
            cover = cover.to(steganogan.device)
            payload = payload.to(steganogan.device)
            
            # Generate the Stego image
            generated = steganogan._encode_step(cover, payload)[0].clamp(-1.0, 1.0).unsqueeze(0)
            
            # Mathematical Loss/Metrics Calculation
            mse = mse_loss(generated, cover)
            total_mse += mse.item()
            total_psnr += calculate_psnr(mse)
            total_ssim += ssim(cover, generated).item()
            success_count += 1
            
        except Exception as e:
            print(f"Skipping {img_name} due to error: {e}")
            continue

    if success_count == 0:
        print("Failed to evaluate any images.")
        return

    avg_mse = total_mse / success_count
    avg_psnr = total_psnr / success_count
    avg_ssim = total_ssim / success_count
    
    print("\n" + "="*40)
    print("      EVALUATION METRICS RESULTS      ")
    print("="*40)
    print(f"Total Images Evaluated: {success_count}")
    print("-" * 40)
    print(f"Mean Squared Error (MSE): {avg_mse:.6f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.2f} dB")
    print(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")
    print("="*40)
    print("Note: High PSNR (>30) and SSIM (~0.99) implies the embedded")
    print("message is visually undetectable to the human eye.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SteGANoGAN Model Metrics")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model weights (.steg)")
    parser.add_argument('--images', type=str, required=True, help="Path to a directory containing evaluation images")
    parser.add_argument('--message', type=str, default="Final Year Project Evaluation Secret", help="Message to embed")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of images to evaluate")
    
    args = parser.parse_args()
    evaluate(args.model, args.images, args.message, limit=args.limit)
