import cv2
import numpy as np

print("OpenCV installed successfully!")
print("Camera test - press 'q' to quit")

# Open webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    print("If you're on a laptop, make sure no other app is using the camera")
    exit()

while True:
    # Read frame from camera
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    
    # Display the frame
    cv2.imshow('Camera Test - Press Q to quit', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

print("Camera test complete! ✅")
print("If you saw yourself, everything works!")