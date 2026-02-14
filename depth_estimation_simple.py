import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print("Loading depth estimation model...")
print("This will download ~100MB on first run - be patient!")

# Load MiDaS depth estimation model
model_type = "MiDaS_small"  # Smaller, faster model for learning
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print("Model loaded! ✅")

# Capture image from camera
print("\nPress SPACE to capture image, ESC to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Position yourself - Press SPACE to capture', frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        print("Cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == 32:  # SPACE
        print("Image captured! Processing...")
        break

cap.release()
cv2.destroyAllWindows()

# Convert BGR to RGB
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Prepare image for model
input_batch = transform(img_rgb)

# Predict depth
print("Estimating depth...")
with torch.no_grad():
    prediction = midas(input_batch)
    # prediction = model(input_batch)
    
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

print("Depth estimation complete! ✅")

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap='plasma')
plt.title('Depth Map (Warm=Close, Cool=Far)')
plt.colorbar(label='Relative Depth')
plt.axis('off')

plt.tight_layout()
plt.savefig('my_first_depth_map.png', dpi=150, bbox_inches='tight')
print("\nSaved result as 'my_first_depth_map.png'")

plt.show()


print("The depth map shows which parts of the scene are close (warm colors) vs far (cool colors)")