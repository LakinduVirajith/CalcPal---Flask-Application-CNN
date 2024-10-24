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
    raise RuntimeError(f"Failed to load the CNN diagnosis model: {str(e)}")

# math symbol CNN
try:
    # Directly provide the path to the model file
    model_path = os.path.join(models_directory, "math-symbol-cnn.h5")
    math_symbol__model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the CNN diagnosis model: {str(e)}")

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


# math symbol CNN
class_names = ['Add', 'Decimal', 'Divide', 'Equal', 'Minus', 'Multiply']

# Preprocessing part
def preprocess_math_image(img_data):
    img = Image.open(io.BytesIO(img_data))

    # Convert the image to RGB if it has an alpha channel (e.g., RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to match the input size of the model
    img = img.resize((150, 150))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image data
    return img_array

@app.route('/predict-symbol', methods=['POST'])
def predict_symbol():
    try:
        # Get the base64 image from the request
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode the base64 image
        image_data = base64.b64decode(data['image'])

        # Preprocess the image
        img_array = preprocess_math_image(image_data)

        # Make a prediction
        prediction = math_symbol__model.predict(img_array)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_names[predicted_class_index]

        # Create a dictionary of class probabilities
        class_probabilities = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}

        # Return the result as JSON
        return jsonify({
            'predicted_class': predicted_class_label,
            'class_probabilities': class_probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
