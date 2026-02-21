import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pillow_heif import register_heif_opener

register_heif_opener()

print("="*70)
print("PROPER DEPTH CALIBRATION SYSTEM")
print("="*70)
print("\nThis uses correct camera geometry for measurements.\n")

# ============================================================================
# Step 1: Estimate Camera Intrinsics
# ============================================================================

print("="*70)
print("STEP 1: Camera Configuration")
print("="*70)

print("\nWe need to estimate your iPhone 13's camera parameters.")
print("Since we don't have EXIF data, we'll use typical values.\n")

# iPhone 13 typical specs
IMAGE_WIDTH = 4032
IMAGE_HEIGHT = 3024
SENSOR_WIDTH_MM = 7.6  # Wide camera sensor
FOCAL_LENGTH_MM = 5.7  # 26mm equivalent

# Calculate focal length in pixels
fx = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_WIDTH
fy = fx  # Assume square pixels
cx = IMAGE_WIDTH / 2
cy = IMAGE_HEIGHT / 2

print(f"Estimated Camera Intrinsics:")
print(f"  Focal length: {fx:.1f} pixels")
print(f"  Principal point: ({cx:.1f}, {cy:.1f})")
print(f"  Image size: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

print("\nCamera Matrix K:")
print(camera_matrix)

# ============================================================================
# Step 2: Load and Process Image
# ============================================================================

print("\n" + "="*70)
print("STEP 2: Load Image and Generate Depth Map")
print("="*70)

image_path = input("\nEnter image path (or press Enter for test_images/IMG_5769.HEIC): ")
if not image_path:
    image_path = "test_images/IMG_5769.HEIC"

print(f"\nLoading: {image_path}")

# Load model
print("Loading MiDaS model...")
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Process image
pil_img = PILImage.open(image_path)
img_rgb = np.array(pil_img.convert('RGB'))
img_height, img_width = img_rgb.shape[:2]

print(f"Image loaded: {img_width} x {img_height}")

# Adjust camera matrix if image size is different
if img_width != IMAGE_WIDTH or img_height != IMAGE_HEIGHT:
    scale_x = img_width / IMAGE_WIDTH
    scale_y = img_height / IMAGE_HEIGHT
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y
    print(f"Adjusted focal length for image size: {fx:.1f} pixels")

input_batch = transform(img_rgb)

print("Estimating depth...")
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()
print("✅ Depth map generated")

# ============================================================================
# Step 3: Calibration with Known Object
# ============================================================================

print("\n" + "="*70)
print("STEP 3: Calibration")
print("="*70)
print("""
For accurate calibration, we need a reference object with KNOWN:
1. Real-world size (width OR height)
2. Its position in the image

Good examples:
  - Door width (typically 0.8-0.9m)
  - Door height (typically 2.0-2.1m)
  - Standard floor tile (0.6m x 0.6m)
  - A4 paper (0.21m x 0.297m)

You'll mark TWO CORNERS of the reference object.
""")

input("\nPress Enter when ready...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(img_rgb)
axes[0].set_title('Mark TWO CORNERS of reference object\n(e.g., top-left and top-right of door)', 
                 fontsize=12, fontweight='bold')
axes[0].axis('off')

im = axes[1].imshow(depth_map, cmap='plasma')
axes[1].set_title('Depth Map', fontsize=12)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

points = []

def onclick(event):
    if event.inaxes == axes[0] and len(points) < 2:
        x, y = int(event.xdata), int(event.ydata)
        
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth_value = depth_map[y, x]
            
            marker = 'ro' if len(points) == 0 else 'go'
            markersize = 12
            
            axes[0].plot(x, y, marker, markersize=markersize)
            axes[1].plot(x, y, marker, markersize=markersize)
            
            label = f"Corner {len(points)+1}"
            color = 'red' if len(points)==0 else 'green'
            axes[0].text(x, y-20, label, color='white', fontsize=11, 
                        fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=color))
            
            points.append({'pixel': (x, y), 'depth_value': depth_value})
            
            print(f"✓ {label}: pixel ({x}, {y}), depth value: {depth_value:.2f}")
            
            if len(points) == 2:
                axes[0].plot([points[0]['pixel'][0], points[1]['pixel'][0]], 
                           [points[0]['pixel'][1], points[1]['pixel'][1]], 
                           'y-', linewidth=3)
                print("\n✅ Both corners marked! Close window to continue.")
            
            fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if len(points) != 2:
    print("\n❌ Need exactly 2 points. Exiting.")
    exit()

# Get real-world size
print("\n" + "="*70)
print("What did you mark?")
print("="*70)
print("1. Door width (horizontal)")
print("2. Door height (vertical)")
print("3. Other object")

choice = input("Enter choice (1/2/3): ")

if choice == "1":
    real_size = float(input("Door width in meters (typically 0.8-0.9): ") or "0.9")
    dimension = "width"
elif choice == "2":
    real_size = float(input("Door height in meters (typically 2.0-2.1): ") or "2.0")
    dimension = "height"
else:
    real_size = float(input("Real size in meters: "))
    dimension = "size"

# ============================================================================
# Step 4: Calculate Scale Factor
# ============================================================================

print("\n" + "="*70)
print("STEP 4: Calculate Calibration")
print("="*70)

# Get pixel coordinates
p1 = points[0]['pixel']
p2 = points[1]['pixel']

# Calculate pixel distance
pixel_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Get average depth at the two points
avg_depth_value = (points[0]['depth_value'] + points[1]['depth_value']) / 2

print(f"\nPixel distance: {pixel_dist:.1f} pixels")
print(f"Average depth value: {avg_depth_value:.2f}")
print(f"Real {dimension}: {real_size} meters")

# The key insight: at a given depth Z, the relationship is:
# real_size = (pixel_size * Z) / f
# 
# We know real_size and pixel_size, so we can solve for Z:
# Z = (real_size * f) / pixel_size

# Use appropriate focal length based on dimension
if abs(p2[1] - p1[1]) > abs(p2[0] - p1[0]):
    # Mostly vertical
    f_to_use = fy
else:
    # Mostly horizontal
    f_to_use = fx

estimated_Z = (real_size * f_to_use) / pixel_dist

print(f"\nUsing focal length: {f_to_use:.1f} pixels")
print(f"Estimated actual depth: {estimated_Z:.2f} meters")

# Now we can find the scale factor
# depth_value * scale = real_depth
scale_factor = estimated_Z / avg_depth_value

print(f"\n🎯 Scale Factor: {scale_factor:.6f} meters per depth unit")
print(f"   (depth_value * {scale_factor:.6f} = real depth in meters)")

# ============================================================================
# Step 5: Interactive Measurement
# ============================================================================

print("\n" + "="*70)
print("STEP 5: Measure Anything!")
print("="*70)
print("""
Now you can:
1. Click single points to measure distance from camera
2. Click two points to measure size of object
3. Right-click to clear last measurement
""")

input("Press Enter to start measuring...")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

axes2[0].imshow(img_rgb)
axes2[0].set_title('LEFT CLICK: measure point | RIGHT CLICK: clear last', 
                  fontsize=12, fontweight='bold')
axes2[0].axis('off')

# Mark calibration points
axes2[0].plot(p1[0], p1[1], 'ro', markersize=8, alpha=0.5, label='Calibration')
axes2[0].plot(p2[0], p2[1], 'go', markersize=8, alpha=0.5)
axes2[0].legend()

im2 = axes2[1].imshow(depth_map, cmap='plasma')
axes2[1].set_title('Depth Map', fontsize=12)
axes2[1].axis('off')
plt.colorbar(im2, ax=axes2[1])

measurements = []
plot_objects = []

def onclick2(event):
    global measurements, plot_objects
    
    if event.inaxes == axes2[0]:
        if event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            
            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                depth_value = depth_map[y, x]
                
                # Convert to real depth
                real_depth = depth_value * scale_factor
                
                # Mark point
                point_plot = axes2[0].plot(x, y, 'c*', markersize=12)[0]
                point_plot2 = axes2[1].plot(x, y, 'c*', markersize=12)[0]
                
                text_plot = axes2[0].text(x, y-25, f'{real_depth:.2f}m', 
                             color='cyan', fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                plot_objects.extend([point_plot, point_plot2, text_plot])
                
                measurements.append({
                    'pixel': (x, y),
                    'depth_value': depth_value,
                    'real_depth': real_depth
                })
                
                print(f"📍 Point {len(measurements)}: ({x}, {y}) → {real_depth:.2f}m from camera")
                
                # If we have 2 points, calculate distance between them
                if len(measurements) >= 2:
                    m1 = measurements[-2]
                    m2 = measurements[-1]
                    
                    # Calculate 3D distance
                    # First, convert pixels to 3D coordinates
                    x1, y1 = m1['pixel']
                    x2, y2 = m2['pixel']
                    Z1 = m1['real_depth']
                    Z2 = m2['real_depth']
                    
                    # Unproject to 3D
                    X1 = (x1 - cx) * Z1 / fx
                    Y1 = (y1 - cy) * Z1 / fy
                    
                    X2 = (x2 - cx) * Z2 / fx
                    Y2 = (y2 - cy) * Z2 / fy
                    
                    # 3D Euclidean distance
                    dist_3d = np.sqrt((X2-X1)**2 + (Y2-Y1)**2 + (Z2-Z1)**2)
                    
                    # Draw line
                    line_plot = axes2[0].plot([x1, x2], [y1, y2], 'y-', linewidth=2)[0]
                    plot_objects.append(line_plot)
                    
                    # Midpoint for label
                    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                    dist_text = axes2[0].text(mid_x, mid_y, f'{dist_3d:.2f}m', 
                                 color='yellow', fontsize=13, fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
                    plot_objects.append(dist_text)
                    
                    print(f"   📏 Distance between last 2 points: {dist_3d:.2f}m")
                
                fig2.canvas.draw()
        
        elif event.button == 3:  # Right click - clear last
            if measurements:
                measurements.pop()
                # Remove last few plot objects
                for _ in range(min(4, len(plot_objects))):
                    if plot_objects:
                        obj = plot_objects.pop()
                        obj.remove()
                fig2.canvas.draw()
                print("↩️  Removed last measurement")

fig2.canvas.mpl_connect('button_press_event', onclick2)
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("MEASUREMENT SUMMARY")
print("="*70)

if measurements:
    print(f"\nTotal measurements: {len(measurements)}")
    print(f"\nScale factor: {scale_factor:.6f} m/unit")
    print(f"Calibration object: {real_size}m {dimension}")
    print(f"\nAll measured points:")
    for i, m in enumerate(measurements, 1):
        x, y = m['pixel']
        print(f"  {i}. ({x:4d}, {y:4d}) → {m['real_depth']:.2f}m from camera")

print("\n✅ Calibration system complete!")
print("\nWhat you learned:")
print("  • Camera intrinsics and their importance")
print("  • Proper geometric calibration")
print("  • Converting depth maps to real measurements")
print("  • 3D distance calculation from 2D images")