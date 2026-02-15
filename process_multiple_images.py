import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image as PILImage
from pillow_heif import register_heif_opener
register_heif_opener()

print("Loading depth estimation model...")
model_type = "DPT_Large"  # More accurate, but slower model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print("Model loaded! ✅\n")

# Get all image files in current directory
image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG','.heic']
image_files = []

for ext in image_extensions:
    image_files.extend(Path('test_images').glob(f'*{ext}'))

# Filter out result images
image_files = [f for f in image_files if 'depth' not in f.name.lower() 
               and 'result' not in f.name.lower()]

if len(image_files) == 0:
    print("No images found in current directory!")
    print("Please add some .jpg or .png images to process.")
    exit()

print(f"Found {len(image_files)} images to process:\n")
for img_file in image_files:
    print(f"  - {img_file.name}")

print("\nProcessing images...\n")

# Create output folder
output_folder = Path("depth_results")
output_folder.mkdir(exist_ok=True)

# Process each image
for i, img_path in enumerate(image_files, 1):
    print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
    
    # Load image (supports HEIC now!)
    try:
        pil_img = PILImage.open(str(img_path))
        img_rgb = np.array(pil_img.convert('RGB'))
    except Exception as e:
        print(f"  ❌ Could not load image: {e}")
        continue
    
    # Prepare for model
    input_batch = transform(img_rgb)
    
    # Predict depth
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original: {img_path.name}', fontsize=12)
    axes[0].axis('off')
    
    im = axes[1].imshow(depth_map, cmap='plasma')
    axes[1].set_title('Depth Map (Warm=Close, Cool=Far)', fontsize=12)
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1], label='Relative Depth', fraction=0.046)
    
    plt.tight_layout()
    
    # Save result
    output_name = output_folder / f"depth_{img_path.stem}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_name}")
    print(f"  📊 Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}\n")

print(f"\n🎉 Complete! Processed {len(image_files)} images")
print(f"📁 Results saved in: {output_folder}/")
print("\nNow check the depth_results folder to see your depth maps!")