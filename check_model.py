import torch
path = r'C:\march\GAN-RGB\SteganoGAN\models\sequential_rgb_FINAL.steg'
try:
    obj = torch.load(path, map_location='cpu')
    print(f"Type: {type(obj)}")
    print(f"Keys if dict: {list(obj.keys()) if isinstance(obj, dict) else 'N/A'}")
except Exception as e:
    print(f"Error: {e}")
