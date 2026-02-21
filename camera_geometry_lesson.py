import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*70)
print("LESSON: HOW CAMERAS SEE THE WORLD")
print("="*70)

# ============================================================================
# PART 1: The Pinhole Camera Model
# ============================================================================

print("\n" + "="*70)
print("PART 1: The Pinhole Camera Model")
print("="*70)

print("""
A camera works like a pinhole:
- Light from 3D world passes through a small point (lens center)
- Projects onto 2D image plane (sensor)
- This creates PERSPECTIVE: far things look smaller

Camera Coordinate System:
    Z-axis: Pointing OUT of camera (depth direction)
    X-axis: Horizontal (left-right)
    Y-axis: Vertical (up-down)
    
Camera is at origin (0, 0, 0)
Everything you see has positive Z (in front of camera)
""")

# Visualize camera geometry
fig = plt.figure(figsize=(14, 10))

# 3D visualization
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Camera at origin
ax1.scatter([0], [0], [0], c='red', s=200, marker='o', label='Camera')

# Viewing direction
ax1.quiver(0, 0, 0, 0, 0, 5, color='red', arrow_length_ratio=0.1, linewidth=3, label='View Direction (Z)')
ax1.quiver(0, 0, 0, 2, 0, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
ax1.quiver(0, 0, 0, 0, 2, 0, color='blue', arrow_length_ratio=0.1, linewidth=2)

# Objects at different depths
objects_3d = [
    (0, 0, 3, 'Close object'),
    (1, 1, 5, 'Medium distance'),
    (-1, -0.5, 8, 'Far object'),
]

for x, y, z, label in objects_3d:
    ax1.scatter([x], [y], [z], s=100, label=label)
    # Draw line from camera to object
    ax1.plot([0, x], [0, y], [0, z], 'k--', alpha=0.3)

ax1.set_xlabel('X (horizontal)')
ax1.set_ylabel('Y (vertical)')
ax1.set_zlabel('Z (depth)')
ax1.set_title('3D World View')
ax1.legend()
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_zlim([0, 10])

# ============================================================================
# PART 2: Perspective Projection
# ============================================================================

print("\n" + "="*70)
print("PART 2: Perspective Projection - The KEY Equation")
print("="*70)

print("""
The FUNDAMENTAL equation of camera projection:

    x_pixel = (f * X) / Z
    y_pixel = (f * Y) / Z
    
Where:
    f = focal length (in pixels)
    X, Y, Z = 3D coordinates of object
    x_pixel, y_pixel = where it appears in image
    
KEY INSIGHT: Notice the division by Z!
- Same object at 2x distance (2Z) appears at HALF the pixel position
- This creates perspective: far things look smaller
- This is why depth is NOT linear!
""")

# Demonstrate perspective
ax2 = fig.add_subplot(2, 2, 2)

focal_length = 500  # pixels (typical for iPhone)

# Same-sized objects at different depths
depths = np.array([3, 5, 8, 12])
object_height_real = 2.0  # meters (like a door)

# Calculate how tall they appear in pixels
pixel_heights = (focal_length * object_height_real) / depths

ax2.bar(depths, pixel_heights, width=0.8, color='skyblue', edgecolor='black')
ax2.set_xlabel('Object Distance (meters)', fontsize=12)
ax2.set_ylabel('Height in Image (pixels)', fontsize=12)
ax2.set_title('Same Object at Different Distances', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

for i, (d, h) in enumerate(zip(depths, pixel_heights)):
    ax2.text(d, h + 10, f'{h:.0f}px', ha='center', fontsize=10, fontweight='bold')

print(f"\nExample: 2-meter tall door at different distances")
print(f"Focal length: {focal_length} pixels")
for d, h in zip(depths, pixel_heights):
    print(f"  At {d}m distance → appears {h:.1f} pixels tall")

# ============================================================================
# PART 3: Depth Map Values vs Real Distance
# ============================================================================

print("\n" + "="*70)
print("PART 3: What Depth Maps Actually Give You")
print("="*70)

print("""
MiDaS depth estimation gives you:
    - RELATIVE depth values (not meters!)
    - Disparity-like values (inverse of depth)
    - Higher value = farther away (usually)
    
The relationship is approximately:
    depth_value ≈ 1 / real_distance  (inverse!)
    
Or sometimes:
    depth_value ≈ real_distance  (depending on model output)
    
This is why simple multiplication didn't work!
""")

ax3 = fig.add_subplot(2, 2, 3)

# Simulate two different depth representations
real_distances = np.linspace(1, 15, 100)

# Option 1: Direct depth (what we assumed)
depth_direct = real_distances

# Option 2: Disparity/inverse depth (what MiDaS often gives)
depth_inverse = 1000 / real_distances  # scaled inverse

ax3.plot(real_distances, depth_direct, 'b-', linewidth=2, label='Direct depth (linear)')
ax3.plot(real_distances, depth_inverse, 'r-', linewidth=2, label='Inverse depth (disparity)')
ax3.set_xlabel('Real Distance (meters)', fontsize=12)
ax3.set_ylabel('Depth Map Value', fontsize=12)
ax3.set_title('Two Ways Depth Can Be Represented', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# PART 4: Why Your Calibration Failed
# ============================================================================

print("\n" + "="*70)
print("PART 4: Why Your Calibration Failed")
print("="*70)

print("""
Your calibration issue:

1. You marked VERTICAL points (top and bottom of door)
   - Both at approximately the SAME depth (Z)
   - Difference in Y (vertical), not Z (depth)
   
2. Then used that scale factor for DEPTH differences
   - Different geometry!
   - Vertical distance ≠ depth distance
   
3. The formula needed:
   For vertical/horizontal at same depth:
       real_size = (depth_map_value * pixel_distance) / focal_length
       
   For depth differences:
       real_depth_diff = f(depth_value_diff, camera_intrinsics)
       
It's like measuring height with a depth ruler!
""")

ax4 = fig.add_subplot(2, 2, 4)

# Illustrate the problem
ax4.text(0.5, 0.8, 'Your Calibration:', ha='center', fontsize=14, 
         fontweight='bold', transform=ax4.transAxes)

ax4.text(0.1, 0.6, '1. Marked door height (vertical)\n   Both points at ~same Z', 
         fontsize=11, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax4.text(0.1, 0.4, '2. Got "scale factor" for that\n   geometry', 
         fontsize=11, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax4.text(0.1, 0.2, '3. Applied to DEPTH differences\n   ❌ Wrong geometry!', 
         fontsize=11, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))

ax4.text(0.5, 0.05, 'Need different calibration for depth vs. size!', 
         ha='center', fontsize=12, fontweight='bold', 
         transform=ax4.transAxes, color='red')

ax4.axis('off')

plt.tight_layout()
plt.savefig('camera_geometry_lesson.png', dpi=150)
print("\n✅ Saved visualization as 'camera_geometry_lesson.png'")
plt.show()

# ============================================================================
# PART 5: The Math You Need
# ============================================================================

print("\n" + "="*70)
print("PART 5: The Correct Math")
print("="*70)

print("""
To convert depth maps to real measurements, you need:

A) CAMERA INTRINSICS (from camera calibration):
   - Focal length (fx, fy) in pixels
   - Principal point (cx, cy) - image center
   - Lens distortion parameters
   
B) DEPTH CALIBRATION:
   - Scale factor (if depth is relative)
   - Zero point (reference distance)
   
C) GEOMETRIC RELATIONSHIPS:
   
   For a 3D point (X, Y, Z) projecting to pixel (u, v):
   
   u = fx * (X/Z) + cx
   v = fy * (Y/Z) + cy
   
   Inverse (pixel to 3D, if you know depth Z):
   
   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy
   Z = depth_value * scale_factor
   
   For object SIZE at known depth:
   
   real_width = (pixel_width * Z) / fx
   real_height = (pixel_height * Z) / fy
""")

# Example calculation
print("\n" + "="*70)
print("EXAMPLE: iPhone 13 Typical Values")
print("="*70)

# iPhone 13 approximate camera specs
image_width = 4032  # pixels
image_height = 3024  # pixels
sensor_width = 7.6  # mm (approximate)
focal_length_mm = 5.7  # mm (approximate wide camera)

# Calculate focal length in pixels
fx = (focal_length_mm / sensor_width) * image_width
fy = fx  # assume square pixels

cx = image_width / 2
cy = image_height / 2

print(f"Image resolution: {image_width} x {image_height}")
print(f"Focal length: {focal_length_mm}mm = {fx:.1f} pixels")
print(f"Principal point: ({cx:.1f}, {cy:.1f})")

print("\nNow let's say you know an object is 3 meters away (Z = 3m)")
print("and it's 500 pixels wide in the image...")

Z = 3.0  # meters
pixel_width = 500

real_width = (pixel_width * Z) / fx
print(f"\nReal width = ({pixel_width} * {Z}) / {fx:.1f}")
print(f"Real width = {real_width:.3f} meters")
print(f"That's {real_width*100:.1f} centimeters!")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. Cameras use PERSPECTIVE PROJECTION (division by depth)
2. Far objects appear smaller (non-linear relationship)
3. Depth maps give RELATIVE depth, not absolute
4. Different formulas for:
   - Object size at known depth
   - Depth differences between objects
   - 3D reconstruction
5. Need camera intrinsics for accurate measurements
6. iPhone has this info! (We'll use it later)

Next: Build a proper calibration system!
""")

print("\n✅ Lesson complete!")