import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("mask_detector.h5")  # Ensure this file exists

def predict_mask(image_path):
    image = cv2.imread(image_path)

    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Could not read the image at {image_path}. Check the file path.")
        return

    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(image)[0][0]
    
    # Print raw prediction to verify
    print(f"Raw Prediction: {prediction}")

    # Adjust threshold based on the raw prediction values
    if prediction < 0.5:
        print("No Mask Detected ❌")
    else:
        print("Mask Detected ✅")

# Provide the absolute image path
image_path = "C:/Users/Sridh/Desktop/Face-Mask-Detection/with_mask/with_mask643.jpeg"
predict_mask(image_path)
