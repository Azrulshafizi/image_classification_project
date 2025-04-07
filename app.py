from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import traceback
from werkzeug.utils import secure_filename
import gc
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Global variable for model
_model = None

def get_model():
    """Lazy-load YOLO model only when needed"""
    global _model
    if _model is None:
        logger.info("Loading YOLO model...")
        try:
            # Import heavy libraries only when needed
            from ultralytics import YOLO
            _model = YOLO("yolov10n.pt")
            
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            _model = None
    return _model

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint to check if the server is running"""
    # Don't load model on health check to save memory
    return jsonify({"status": "ok"})

@app.route("/live", methods=["GET"])
def live_detection():
    """Live real-time object detection with YOLO"""
    return render_template("live_camera.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if image exists in request
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Read image from request
        file = request.files["image"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Check if the file is an allowed image type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save file temporarily
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read image with OpenCV
            image = cv2.imread(filepath)
            
            if image is None:
                return jsonify({"error": "Could not decode image after saving"}), 400
                
        except Exception as save_error:
            return jsonify({"error": f"Error saving or reading image: {str(save_error)}"}), 400

        # Load model and perform prediction
        try:
            # Get model (lazy loading)
            model = get_model()
            
            # Check if model is loaded
            if model is None:
                return jsonify({"error": "Model not loaded. Check server logs."}), 500
            
            # Log memory usage before prediction
            log_memory_usage("Before prediction")
            
            # Perform YOLO prediction
            results = model(image, device="cpu")
            
            # Log memory usage after prediction
            log_memory_usage("After prediction")
            
            if results is None or len(results) == 0:
                return jsonify({"predictions": []}), 200
                
            detections = results[0].boxes  # Get detected objects

            # Format results
            predictions = []
            for box in detections:
                try:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])  # Convert to float first
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Then convert to int
                    class_id = int(box.cls[0])  # Class ID
                    confidence = float(box.conf[0])  # Confidence score
                    predictions.append({"class": class_id, "confidence": confidence, "bbox": [x1, y1, x2, y2]})
                except Exception as box_error:
                    logger.error(f"Error processing detection box: {str(box_error)}")
                    continue

            # Clean up the temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
                
            # Force garbage collection after processing
            gc.collect()
            
            return jsonify({"predictions": predictions})
            
        except Exception as model_error:
            error_details = traceback.format_exc()
            logger.error(f"Error during object detection: {str(model_error)}")
            logger.error(error_details)
            return jsonify({"error": f"Error during object detection: {str(model_error)}"}), 500
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(error_details)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def log_memory_usage(tag=""):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage {tag}: {mb:.2f} MB")
    except ImportError:
        logger.info(f"Memory usage {tag}: psutil not available")

if __name__ == "__main__":
    # Use 0.0.0.0 to listen on all interfaces
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))