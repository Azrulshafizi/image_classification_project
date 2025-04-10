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
from functools import lru_cache

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
            
            # Use a smaller model for faster inference
            # Options: yolov8n (fastest), yolov8s, yolov8m, yolov8l, yolov8x (most accurate)
            _model = YOLO("yolov8n.pt")
            
            # Check for GPU availability
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
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

# Image preprocessing cache - stores recently processed images to avoid duplicated work
@lru_cache(maxsize=32)
def preprocess_image(image_hash):
    """This is a placeholder for the actual cache implementation.
    In a real implementation, we would store preprocessed images by hash."""
    pass

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
                
            # OPTIMIZATION: More aggressive resizing for faster processing
            # YOLOv8 works well with 416x416 or 320x320 for faster inference
            target_size = 320  # Smaller size for faster inference
            h, w = image.shape[:2]
            scale = target_size / max(h, w)
            if max(h, w) > target_size:
                new_w, new_h = int(w * scale), int(h * scale)
                logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Use INTER_AREA for downsampling
                
        except Exception as save_error:
            return jsonify({"error": f"Error saving or reading image: {str(save_error)}"}), 400

        # Load model and perform prediction
        try:
            # Get model (lazy loading)
            model = get_model()
            
            # Check if model is loaded
            if model is None:
                return jsonify({"error": "Model not loaded. Check server logs."}), 500
            
            # Check for GPU availability
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # Log memory usage before prediction
            log_memory_usage("Before prediction")
            
            # OPTIMIZATION: Run optimized inference
            # Higher confidence threshold further reduces processing by eliminating weak detections early
            # Using half-precision for faster inference on supported GPUs
            results = model(image, 
                           device=device,
                           conf=0.35,  # Higher confidence threshold
                           iou=0.45,   # Higher IOU threshold for NMS (fewer overlapping boxes)
                           half=True,  # Use half precision (FP16) for faster inference
                           max_det=20) # Limit detections to top 20 for faster processing
            
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
                    
                    # If image was resized, adjust bounding box coordinates back to original scale
                    if max(h, w) > target_size:
                        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                        
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Then convert to int
                    class_id = int(box.cls[0])  # Class ID
                    confidence = float(box.conf[0])  # Confidence score
                    
                    # Get class name (optional)
                    class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"
                    
                    predictions.append({
                        "class": class_id, 
                        "class_name": class_name,
                        "confidence": confidence, 
                        "bbox": [x1, y1, x2, y2]
                    })
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

# Add a new endpoint for batch processing multiple images
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        # Check if images exist in request
        if "images" not in request.files:
            return jsonify({"error": "No images uploaded"}), 400

        # Get all files
        files = request.files.getlist("images")
        if not files or len(files) == 0:
            return jsonify({"error": "No files selected"}), 400
            
        # Process each image
        batch_results = []
        image_batch = []
        filepaths = []
        scales = []
        original_sizes = []
        
        for file in files:
            # Check if the file is an allowed image type
            allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                continue  # Skip invalid files
                
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filepaths.append(filepath)
            
            # Read image
            image = cv2.imread(filepath)
            if image is None:
                continue  # Skip invalid images
                
            # Resize image
            target_size = 320
            h, w = image.shape[:2]
            original_sizes.append((w, h))
            scale = target_size / max(h, w)
            scales.append(scale)
            
            if max(h, w) > target_size:
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
            image_batch.append(image)
        
        # If no valid images were found, return an error
        if not image_batch:
            return jsonify({"error": "No valid images found in the batch"}), 400
            
        # Get model
        model = get_model()
        if model is None:
            return jsonify({"error": "Model not loaded. Check server logs."}), 500
            
        # Check for GPU availability
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Perform batch prediction - much faster than individual predictions
        results = model(image_batch, 
                       device=device,
                       conf=0.35,
                       iou=0.45,
                       half=True,
                       max_det=20)
        
        # Process results for each image
        for i, result in enumerate(results):
            predictions = []
            detections = result.boxes
            
            for box in detections:
                try:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    
                    # Adjust coordinates back to original scale
                    orig_w, orig_h = original_sizes[i]
                    scale = scales[i]
                    
                    if max(orig_h, orig_w) > target_size:
                        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                        
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get class name
                    class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"
                    
                    predictions.append({
                        "class": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
                except Exception:
                    continue
                    
            batch_results.append({
                "filename": os.path.basename(filepaths[i]),
                "predictions": predictions
            })
        
        # Clean up temporary files
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
                
        # Force garbage collection
        gc.collect()
        
        return jsonify({"results": batch_results})
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error in batch processing: {str(e)}")
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
    # Check for GPU support on startup and print info
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            # Pre-warm the GPU by allocating a small tensor
            torch.cuda.empty_cache()
            dummy = torch.zeros(1).cuda()
            del dummy
            # Set environment variables for better CUDA performance
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['OMP_NUM_THREADS'] = '1'
        else:
            logger.info("No GPU available, using CPU")
    except:
        logger.info("Error checking for GPU, assuming CPU only")
    
    # Use 0.0.0.0 to listen on all interfaces
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))