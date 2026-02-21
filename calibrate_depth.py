
# NOTE: This calibration approach is INCORRECT
# It doesn't account for camera geometry properly
# See proper_calibration.py for the correct approach
# Keeping this to show the learning process

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pillow_heif import register_heif_opener

register_heif_opener()

print("="*60)
print("DEPTH CALIBRATION SYSTEM")
print("="*60)
print("\nThis tool helps convert relative depth to real measurements.")
print("You'll mark two points and tell us the real distance between them.\n")

# Load model
print("Loading model...")
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
print("Model loaded! ✅\n")

# Get image
image_path = input("Enter image path (or press Enter for test_images/IMG_5769.HEIC): ")
if not image_path:
    image_path = "test_images/IMG_5769.HEIC"

# Process image
print(f"\nProcessing: {image_path}")
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
print("✅ Depth map generated\n")

# Interactive calibration
print("="*60)
print("STEP 1: Mark two points with KNOWN distance between them")
print("="*60)
print("Examples:")
print("  - Top and bottom of a door (usually ~2 meters)")
print("  - Width of a door frame (~0.9 meters)")
print("  - Height of a person you know")
print("  - Floor tiles (if you know size)")
print("\nClick two points, then close the window")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(img_rgb)
axes[0].set_title('Click TWO points with known distance', fontsize=14, fontweight='bold')
axes[0].axis('off')

im = axes[1].imshow(depth_map, cmap='plasma')
axes[1].set_title('Depth Map', fontsize=14)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

points = []

def onclick(event):
    if event.inaxes == axes[0] and len(points) < 2:
        x, y = int(event.xdata), int(event.ydata)
        
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth_value = depth_map[y, x]
            
            # Mark point
            marker = 'ro' if len(points) == 0 else 'go'
            axes[0].plot(x, y, marker, markersize=15)
            axes[1].plot(x, y, marker, markersize=15)
            
            label = f"Point {len(points)+1}"
            axes[0].text(x, y-20, label, color='white', fontsize=12, 
                        fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='red' if len(points)==0 else 'green'))
            
            points.append((x, y, depth_value))
            
            print(f"✓ {label}: ({x}, {y}) → Depth value: {depth_value:.2f}")
            
            if len(points) == 2:
                # Draw line between points
                axes[0].plot([points[0][0], points[1][0]], 
                           [points[0][1], points[1][1]], 
                           'y-', linewidth=3)
                axes[1].plot([points[0][0], points[1][0]], 
                           [points[0][1], points[1][1]], 
                           'y-', linewidth=3)
                print("\n✅ Both points marked! Close the window to continue.")
            
            fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if len(points) != 2:
    print("\n❌ Need exactly 2 points. Exiting.")
    exit()

# Get real distance
print("\n" + "="*60)
print("STEP 2: Enter the REAL distance between those points")
print("="*60)

real_distance = float(input("Distance in METERS (e.g., 2.0 for a door height): "))

# Calculate calibration
depth_diff = abs(points[1][2] - points[0][2])
pixel_dist = np.sqrt((points[1][0] - points[0][0])**2 + 
                     (points[1][1] - points[0][1])**2)

print(f"\nDepth difference: {depth_diff:.2f}")
print(f"Pixel distance: {pixel_dist:.1f} pixels")
print(f"Real distance: {real_distance} meters")

# Calculate scale factor (simplified - assumes points at similar depth)
avg_depth = (points[0][2] + points[1][2]) / 2
scale_factor = real_distance / depth_diff

print(f"\n🎯 Calibration factor: {scale_factor:.6f} meters per depth unit")

# Now you can measure anything!
print("\n" + "="*60)
print("STEP 3: Measure anything in the image!")
print("="*60)
print("Click on any object to estimate its distance from camera")
print("Close window when done")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

axes2[0].imshow(img_rgb)
axes2[0].set_title('Click to measure distance from camera', fontsize=14, fontweight='bold')
axes2[0].axis('off')

# Mark calibration points
axes2[0].plot(points[0][0], points[0][1], 'ro', markersize=10, label='Cal Point 1')
axes2[0].plot(points[1][0], points[1][1], 'go', markersize=10, label='Cal Point 2')
axes2[0].legend()

im2 = axes2[1].imshow(depth_map, cmap='plasma')
axes2[1].set_title('Depth Map', fontsize=14)
axes2[1].axis('off')
plt.colorbar(im2, ax=axes2[1])

measurements = []

def onclick2(event):
    if event.inaxes == axes2[0]:
        x, y = int(event.xdata), int(event.ydata)
        
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth_value = depth_map[y, x]
            
            # Rough distance estimate (simplified)
            estimated_distance = depth_value * scale_factor
            
            # Mark point
            axes2[0].plot(x, y, 'y*', markersize=15)
            axes2[1].plot(x, y, 'y*', markersize=15)
            
            axes2[0].text(x, y-30, f'{estimated_distance:.2f}m', 
                         color='yellow', fontsize=14, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            fig2.canvas.draw()
            
            print(f"📏 Point at ({x}, {y}): ~{estimated_distance:.2f} meters from camera")
            measurements.append((x, y, estimated_distance))

fig2.canvas.mpl_connect('button_press_event', onclick2)
plt.show()

# Summary
if measurements:
    print("\n" + "="*60)
    print("MEASUREMENT SUMMARY")
    print("="*60)
    for i, (x, y, dist) in enumerate(measurements, 1):
        print(f"Point {i}: ({x:4d}, {y:4d}) → {dist:.2f} meters")
    
    if len(measurements) > 1:
        print("\n📐 Distances between measured points:")
        for i in range(len(measurements)-1):
            d1 = measurements[i][2]
            d2 = measurements[i+1][2]
            diff = abs(d2 - d1)
            print(f"   Point {i+1} to Point {i+2}: {diff:.2f} meters")

print("\n✅ Calibration complete!")
print(f"💾 Scale factor: {scale_factor:.6f} meters/unit")
print("\nYou can now use this factor to convert any depth value to meters!")