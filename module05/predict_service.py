from flask import Flask, request, jsonify
import pickle
import numpy as np
from typing import List, Dict, Union
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    with open('model1.bin', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found! Ensure model1.bin exists")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

try:
    with open('dv.bin', 'rb') as f:
        dv = pickle.load(f)
    logger.info("DictVectorizer loaded successfully")
except FileNotFoundError:
    logger.error("DictVectorizer file not found! Ensure dv.bin exists")
    dv = None
except Exception as e:
    logger.error(f"Error loading DictVectorizer: {str(e)}")
    dv = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions
    Expected JSON format:
    {
        "job": string, "duration": int, "poutcome": string
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        features = dv.transform(data)

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "dv_loaded": dv is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
