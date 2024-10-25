from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from keras.layers import DepthwiseConv2D
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import base64
from PIL import Image, ImageOps
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

def custom_depthwise_conv2d(*args, **kwargs):
    # You can customize how to handle parameters or ignore them if necessary
    kwargs.pop('groups', None)  # Remove unrecognized parameters
    return DepthwiseConv2D(*args, **kwargs)

# Load models
models_directory = os.path.join(os.path.dirname(__file__), 'models')
models = {
    "number": load_model(os.path.join(models_directory, "number_cnn_model.h5")),
    "symbol": load_model(os.path.join(models_directory, "symbol_cnn_model.h5")),
    "shape": load_model(os.path.join(models_directory, "shape_cnn_model.h5"), custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
}

# Preprocessing functions
def preprocess_image(img, size=(28, 28), normalize=True):
    img = img.resize(size, Image.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0 if normalize else img_array

# Routes
@app.route('/')
def home():
    return "Welcome to the Dyscalculia Prediction App!"

@app.route('/predict-number', methods=['POST'])
def predict_number():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image provided'}), 400
    
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert('L')
    prediction = models["number"].predict(preprocess_image(img))
    return jsonify({'predicted_label': int(np.argmax(prediction))})

@app.route('/predict-symbol', methods=['POST'])
def predict_symbol():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    class_names = ['Add', 'Decimal', 'Divide', 'Equal', 'Minus', 'Multiply']
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')
    prediction = models["symbol"].predict(preprocess_image(img, size=(150, 150)))
    probabilities = {class_names[i]: float(pred) for i, pred in enumerate(prediction[0])}
    return jsonify({'predicted_class': class_names[np.argmax(prediction)], 'class_probabilities': probabilities})

@app.route('/predict-shape', methods=['POST'])
def predict_shape():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    class_names = ['Circle', 'Square', 'Triangle']
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)

    # Ensure the shape is (1, 224, 224, 3) by expanding the dimensions
    img_array = np.expand_dims(np.array(img).astype(np.float32) / 127.5 - 1, axis=0)
    
    # Perform prediction
    prediction = models["shape"].predict(img_array)
    return jsonify({
        "class": class_names[np.argmax(prediction)],
        "confidence": float(np.max(prediction))
    })


@app.route('/routes')
def list_routes():
    routes = [{'endpoint': rule.endpoint, 'methods': list(rule.methods - {'HEAD', 'OPTIONS'}), 'url': rule.rule}
              for rule in app.url_map.iter_rules() if rule.endpoint != 'static']
    return jsonify({'status': 'success', 'message': 'Available routes', 'routes': sorted(routes, key=lambda x: x['url'])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
