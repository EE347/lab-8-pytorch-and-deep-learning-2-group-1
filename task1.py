import cv2
import numpy as np
import os
from picamera2 import Picamera2
import time

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directories for saving images
train_dir = "data/train"
test_dir = "data/test"

# Ensure the directories exist
os.makedirs(f"{train_dir}/0", exist_ok=True)
os.makedirs(f"{train_dir}/1", exist_ok=True)
os.makedirs(f"{test_dir}/0", exist_ok=True)
os.makedirs(f"{test_dir}/1", exist_ok=True)

# Function to capture images
def capture_images(label, num_images=60):
    images_captured = 0
    while images_captured < num_images:
        # Capture an image from Picamera2
        image = picam2.capture_array()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If faces are detected, crop and save them
        for (x, y, w, h) in faces:
            face_crop = image[y:y+h, x:x+w]

            # Resize the face crop to 64x64 pixels
            face_resized = cv2.resize(face_crop, (64, 64))

            # Save the image to the appropriate directory
            if images_captured < 50:
                save_dir = f"{train_dir}/{label}"
            else:
                save_dir = f"{test_dir}/{label}"

            image_filename = f"{save_dir}/{images_captured + 1}.jpg"
            cv2.imwrite(image_filename, face_resized)
            images_captured += 1

            # Break the loop if we've captured enough images
            if images_captured >= num_images:
                break

        # Wait a bit before capturing the next image
        time.sleep(0.1)  # Optional small delay to reduce CPU load (optional)

    print(f"Captured {images_captured} images for label {label}.")

#Capture images for teammate 1 (label 0)
print("Capturing images for teammate 1...")
capture_images(label=0)

# Capture images for teammate 2 (label 1)
print("Capturing images for teammate 2...")
capture_images(label=1)

# Stop the camera
picam2.stop()

print("Image capture complete!")
