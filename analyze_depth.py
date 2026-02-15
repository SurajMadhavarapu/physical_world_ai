import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PILImage
from pillow_heif import register_heif_opener

register_heif_opener()

print("Loading depth estimation model...")
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print("Model loaded! ✅\n")

# Let's analyze one of your best images - the doorway scene
image_path = "test_images/IMG_5769.HEIC"  # The doorway image

print(f"Analyzing: {image_path}\n")

# Load and process
pil_img = PILImage.open(image_path)
img_rgb = np.array(pil_img.convert('RGB'))

input_batch = transform(img_rgb)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

# Analysis
print("=== DEPTH STATISTICS ===")
print(f"Min depth: {depth_map.min():.2f}")
print(f"Max depth: {depth_map.max():.2f}")
print(f"Mean depth: {depth_map.mean():.2f}")
print(f"Median depth: {np.median(depth_map):.2f}")
print(f"Std deviation: {depth_map.std():.2f}")

# Find closest and farthest points
min_loc = np.unravel_index(depth_map.argmin(), depth_map.shape)
max_loc = np.unravel_index(depth_map.argmax(), depth_map.shape)

print(f"\nClosest point at pixel: {min_loc} (row, col)")
print(f"Farthest point at pixel: {max_loc} (row, col)")

# Create detailed visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Original image
axes[0, 0].imshow(img_rgb)
axes[0, 0].plot(min_loc[1], min_loc[0], 'r*', markersize=15, label='Closest')
axes[0, 0].plot(max_loc[1], max_loc[0], 'b*', markersize=15, label='Farthest')
axes[0, 0].set_title('Original Image with Extreme Points')
axes[0, 0].legend()
axes[0, 0].axis('off')

# Depth map
im1 = axes[0, 1].imshow(depth_map, cmap='plasma')
axes[0, 1].plot(min_loc[1], min_loc[0], 'r*', markersize=15)
axes[0, 1].plot(max_loc[1], max_loc[0], 'b*', markersize=15)
axes[0, 1].set_title('Depth Map')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])

# Depth histogram
axes[1, 0].hist(depth_map.flatten(), bins=100, color='skyblue', edgecolor='black')
axes[1, 0].set_xlabel('Depth Value')
axes[1, 0].set_ylabel('Number of Pixels')
axes[1, 0].set_title('Depth Distribution')
axes[1, 0].axvline(depth_map.mean(), color='red', linestyle='--', label='Mean')
axes[1, 0].axvline(np.median(depth_map), color='green', linestyle='--', label='Median')
axes[1, 0].legend()

# Horizontal slice analysis (middle row)
middle_row = depth_map.shape[0] // 2
depth_slice = depth_map[middle_row, :]

axes[1, 1].plot(depth_slice, linewidth=2)
axes[1, 1].set_xlabel('Horizontal Position (pixels)')
axes[1, 1].set_ylabel('Depth Value')
axes[1, 1].set_title(f'Depth Profile (Row {middle_row})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('depth_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved detailed analysis as 'depth_analysis.png'")
plt.show()

# Interactive point selection
print("\n" + "="*50)
print("INTERACTIVE MODE")
print("="*50)
print("Click on the original image to see depth at that point")
print("Close the window when done")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

axes2[0].imshow(img_rgb)
axes2[0].set_title('Click on image to query depth')
axes2[0].axis('off')

im2 = axes2[1].imshow(depth_map, cmap='plasma')
axes2[1].set_title('Depth Map')
axes2[1].axis('off')
plt.colorbar(im2, ax=axes2[1])

clicked_points = []

def onclick(event):
    if event.inaxes == axes2[0]:
        x, y = int(event.xdata), int(event.ydata)
        
        # Get depth at clicked point
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth_value = depth_map[y, x]
            
            # Plot point
            axes2[0].plot(x, y, 'ro', markersize=10)
            axes2[1].plot(x, y, 'ro', markersize=10)
            
            # Add text
            axes2[0].text(x, y-20, f'{depth_value:.1f}', 
                         color='red', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            fig2.canvas.draw()
            
            print(f"Clicked at ({x}, {y}): Depth = {depth_value:.2f}")
            clicked_points.append((x, y, depth_value))

fig2.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Summary
if len(clicked_points) > 1:
    print("\n=== CLICKED POINTS SUMMARY ===")
    for i, (x, y, d) in enumerate(clicked_points, 1):
        print(f"Point {i}: ({x}, {y}) → Depth: {d:.2f}")
    
    # Calculate relative distances
    print("\n=== RELATIVE DISTANCES ===")
    for i in range(len(clicked_points)-1):
        d1 = clicked_points[i][2]
        d2 = clicked_points[i+1][2]
        ratio = d2 / d1
        print(f"Point {i+1} vs Point {i+2}: {ratio:.2f}x farther")

print("\n🎉 Analysis complete!")