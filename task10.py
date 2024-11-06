import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from picamera2 import Picamera2
import time

# Function to apply log-softmax
def apply_log_softmax(outputs):
    return torch.nn.functional.log_softmax(outputs, dim=1)

# Function to load the trained model
def load_model(model_path, device):
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Function to preprocess the image before passing it to the model
def preprocess_image(image, device):
    # Convert the image to RGB if it's in RGBA (4 channels)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # Drop the alpha channel

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL image
        transforms.Resize((64, 64)),  # Resize to match the model input size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on pre-trained model
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return image_tensor

# Function to draw the bounding box and label on the image
def draw_label(frame, x, y, w, h, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model (from Task 7)
    model_path = 'lab8/best_model_nll.pth'  # Update with the correct model path
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Error: Cascade Classifier failed to load")
        exit()

    # Initialize Picamera2
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))  # Configure preview size
        picam2.start()
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        exit()

    print("Ready to capture face! Press Enter to capture.")

    try:
        while True:
            # Capture the image from the camera
            frame = picam2.capture_array()

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Crop the face from the image
                face = frame[y:y + h, x:x + w]

                # Preprocess the face image
                face_tensor = preprocess_image(face, device)

                # Perform inference
                with torch.no_grad():
                    outputs = model(face_tensor)
                    _, predicted = torch.max(outputs, 1)

                # Map the predicted class to teammate's name
                if predicted.item() == 0:
                    label = "Teammate 1"
                else:
                    label = "Teammate 2"

                # Draw bounding box and label on the frame
                draw_label(frame, x, y, w, h, label)

            # Display the resulting frame with bounding boxes
            cv2.imshow('Teammate Detection', frame)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCapture process interrupted.")
    finally:
        # Release the camera and close the window
        picam2.stop()
        cv2.destroyAllWindows()
