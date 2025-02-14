from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load your trained model
model = load_model("mask_detector.h5")

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_mask(image_path):
    image = cv2.imread(image_path)

    # Check if the image was loaded properly
    if image is None:
        return "Error: Could not read the image."

    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(image)[0][0]

    # Return result
    if prediction >= 0.5:
        return "No Mask Detected ❌"
    else:
        return "Mask Detected ✅"

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict the mask status
        result = predict_mask(filepath)
        return jsonify({"prediction": result})
    
    return jsonify({"message": "Invalid file type"})

if __name__ == '__main__':
    app.run(debug=True)

