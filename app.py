from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = 'model/TheModel.h5'
model = load_model(model_path)

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the size your model expects
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        # Save the image to the uploads directory
        filepath = os.path.join('uploads', image_file.filename)
        image_file.save(filepath)

        # Load the image and preprocess it
        image = Image.open(filepath)
        image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming a classification model

        # Delete the uploaded file after prediction
        os.remove(filepath)

        # Return the result as JSON
        return jsonify({'prediction': str(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
