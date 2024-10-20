from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import base64
from PIL import Image
import io

# Set up path to the models directory
models_directory = os.path.join(os.path.dirname(__file__), 'models')

# Load the pre-trained model
try:
    # Directly provide the path to the model file
    model_path = os.path.join(models_directory, "predict-number_cnn_model.h5")
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the verbal diagnosis model: {str(e)}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a function to preprocess the image
def preprocess_image(img):
    img = img.resize((28, 28), Image.LANCZOS)  # Resize image
    img = image.img_to_array(img)  # Convert image to array
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 28, 28, 1)
    img = img / 255.0  # Normalize pixel values
    return img

# Define the home route
@app.route('/')
def home():
    """
    Simple home route to check if the server is running.
    """
    return "Welcome to the Dyscalculia CNN Prediction Flask App!"

# Define an API endpoint for prediction using a Base64 string
@app.route('/predict-number', methods=['POST'])
def predict():
    data = request.json
    
    if 'image' not in data:
        return jsonify({'error': 'No image part in the request'}), 400
    
    # Get the Base64 string
    base64_string = data['image']
    
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(img_data)).convert('L')  # Convert to grayscale
    
    # Preprocess the image and make prediction
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_label = np.argmax(predictions, axis=1)
    
    return jsonify({'predicted_label': int(predicted_label[0])})

# Lists all available routes in the Flask application
@app.route('/routes')
def list_routes():
    """
    Endpoint to list all available routes in the Flask application.
    Provides route details including endpoint, methods, and URL.
    """
    routes = [
        {
            'endpoint': rule.endpoint,
            'methods': list(rule.methods - {'HEAD', 'OPTIONS'}),
            'url': rule.rule
        }
        for rule in app.url_map.iter_rules()
        if rule.endpoint != 'static'  # Exclude static files
    ]
    
    # Sort routes by URL for better readability
    routes.sort(key=lambda x: x['url'])
    
    # Create a formatted JSON response
    response = {
        'status': 'success',
        'message': 'List of available routes',
        'routes': routes
    }
    
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)