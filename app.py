from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import requests
import logging

# Flask setup
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Logging setup
logging.basicConfig(level=logging.INFO)

# Model config
MODEL_URL = "https://emotionimageclassifer-model-storage.s3.us-east-2.amazonaws.com/happy_sad_model.h5"
MODEL_PATH = "happy_sad_model.h5"

# Download model from S3
def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("Model file not found locally. Downloading from S3...")
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            logging.info("Model downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise

# Load model
try:
    download_model()
    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Preprocess image
def preprocess_image(file):
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received request to /predict")

    if model is None:
        logging.error("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if 'file' not in request.files:
            logging.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400

        img_array = preprocess_image(file)
        logging.info("Image preprocessed successfully")

        predictions = model.predict(img_array)
        score = float(predictions[0][0])
        logging.info(f"Prediction score: {score}")

        return jsonify({"score": score})

    except Exception as e:
        logging.error(f"Exception in /predict: {e}")
        return jsonify({"error": str(e)}), 500

# Health check
@app.route("/healthz", methods=["GET"])
def health_check():
    return "OK", 200

# Required for Render
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
