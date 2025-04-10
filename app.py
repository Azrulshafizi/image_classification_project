from flask import Flask, render_template, request, jsonify, g
from flask_cors import CORS
import cv2
import numpy as np
import os
import traceback
import time
from werkzeug.utils import secure_filename
import gc
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure upload folder - use /tmp for Render
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir())
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Environment variables with defaults
TARGET_SIZE = int(os.environ.get('TARGET_SIZE', 320))
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.35))
IOU_THRESHOLD = float(os.environ.get('IOU_THRESHOLD', 0.45))
MAX_DETECTIONS = int(os.environ.get('MAX_DETECTIONS', 20))
MODEL_PATH = os.environ.get('MODEL_PATH', 'yolov8n.pt')

# Initialize thread pool for preprocessing
executor = ThreadPoolExecutor(max_workers=2)

# Global variable for model
_model = None

def get_model():
    """Lazy-load YOLO model only when needed"""
    global _model
    if _model is None:
        logger.info(f"Loading YOLO model from {MODEL_PATH}...")
        try:
            # Import heavy libraries only when needed
            from ultralytics import YOLO
            start_time = time.time()
            _model = YOLO(MODEL_PATH)
            load_time = time.time() - start_time
            logger.info(f"YOLO model loaded successfully in {load_time:.2f} seconds")
            
            # Check for GPU availability
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                if device == "cpu":
                    logger.info("No GPU available. Detection will be slower.")
            except ImportError:
                logger.info("PyTorch not properly installed. Using CPU only.")
                
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            _model = None
    return _model

def compress_image(image, quality=75):
    """Compress image to reduce memory usage and processing time"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def process_image(filepath):
    """Process image for prediction - can be run in a thread"""
    try:
        # Read image with OpenCV
        image = cv2.imread(filepath)
        
        if image is None:
            return None, None, None
            
        # Record original dimensions
        h, w = image.shape[:2]
        
        # Compress image
        image = compress_image(image)
        
        # Resize image to target size for faster processing
        scale = TARGET_SIZE / max(h, w)
        if max(h, w) > TARGET_SIZE:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        return image, scale, (h, w)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None, None

def cleanup_resources():
    """Force cleanup of resources to prevent memory leaks"""
    try:
        # Clean up CUDA memory if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    # Force garbage collection
    gc.collect()

@app.before_request
def before_request():
    # Start timer for performance monitoring
    g.start_time = time.time()
    log_memory_usage("Before request")

@app.after_request
def after_request(response):
    # Log request time
    if hasattr(g, 'start_time'):
        elapsed_time = time.time() - g.start_time
        logger.info(f"Request processed in {elapsed_time:.4f} seconds")
    
    # Clean up resources
    log_memory_usage("After request")
    cleanup_resources()
    
    return response

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint to check if the server is running"""
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "memory": get_memory_usage()
    })

@app.route("/warmup", methods=["GET"])
def warmup():
    """Endpoint to keep the service warm and load model"""
    model = get_model()
    status = "ready" if model is not None else "error"
    
    try:
        # Create a small test image and run inference to warm up the model
        test_image = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        if model is not None:
            # Use a low confidence to avoid any actual computation
            model(test_image, conf=0.01, max_det=1)
        
        return jsonify({
            "status": status,
            "model_loaded": model is not None,
            "timestamp": time.time(),
            "memory": get_memory_usage()
        })
    except Exception as e:
        logger.error(f"Error during warmup: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        })

@app.route("/live", methods=["GET"])
def live_detection():
    """Live real-time object detection with YOLO"""
    return render_template("live_camera.html")

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        # Check if image exists in request
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Read image from request
        file = request.files["image"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Check if the file is an allowed image type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save file temporarily
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Image saved to {filepath}")
            
            # Process image (resize, compress)
            image, scale, original_size = process_image(filepath)
            
            if image is None:
                return jsonify({"error": "Could not decode image"}), 400
                
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
            device = "cpu"
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                pass
            
            # Log processing time
            preprocess_time = time.time() - start_time
            logger.info(f"Preprocessing completed in {preprocess_time:.4f} seconds")
            
            # Log memory usage before prediction
            log_memory_usage("Before prediction")
            
            # Start timing inference
            inference_start = time.time()
            
            # Perform YOLO prediction with optimized parameters
            results = model(
                image, 
                device=device,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                half=True,  # Use half precision (FP16) for faster inference
                max_det=MAX_DETECTIONS
            )
            
            # Calculate inference time
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.4f} seconds")
            
            # Log memory usage after prediction
            log_memory_usage("After prediction")
            
            if results is None or len(results) == 0:
                # Clean up the temporary file
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({"predictions": []}), 200
                
            detections = results[0].boxes  # Get detected objects

            # Format results
            predictions = []
            original_h, original_w = original_size
            
            for box in detections:
                try:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])  # Convert to float first
                    
                    # If image was resized, adjust bounding box coordinates back to original scale
                    if scale < 1:  # Only rescale if we actually resized the image
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
                
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Return results with timing information
            return jsonify({
                "predictions": predictions,
                "timing": {
                    "preprocessing": preprocess_time,
                    "inference": inference_time,
                    "total": total_time
                },
                "image_info": {
                    "original_size": [original_w, original_h],
                    "processed_size": [image.shape[1], image.shape[0]]
                }
            })
            
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

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    start_time = time.time()
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
        valid_indices = []
        
        for i, file in enumerate(files):
            # Check if the file is an allowed image type
            allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                continue  # Skip invalid files
                
            # Save file temporarily
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filepaths.append(filepath)
                
                # Process image in thread
                future = executor.submit(process_image, filepath)
                image, scale, original_size = future.result()
                
                if image is None:
                    continue  # Skip invalid images
                    
                image_batch.append(image)
                scales.append(scale)
                original_sizes.append(original_size)
                valid_indices.append(i)
                
            except Exception as save_error:
                logger.error(f"Error processing image {file.filename}: {str(save_error)}")
                continue
        
        # If no valid images were found, return an error
        if not image_batch:
            return jsonify({"error": "No valid images found in the batch"}), 400
            
        # Get model
        model = get_model()
        if model is None:
            return jsonify({"error": "Model not loaded. Check server logs."}), 500
            
        # Check for GPU availability
        device = "cpu"
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        
        # Log preprocessing time
        preprocess_time = time.time() - start_time
        logger.info(f"Batch preprocessing completed in {preprocess_time:.4f} seconds")
        
        # Perform batch prediction
        inference_start = time.time()
        results = model(
            image_batch, 
            device=device,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            half=True,
            max_det=MAX_DETECTIONS
        )
        inference_time = time.time() - inference_start
        logger.info(f"Batch inference completed in {inference_time:.4f} seconds")
        
        # Process results for each image
        for i, result in enumerate(results):
            predictions = []
            detections = result.boxes
            
            # Get original dimensions
            original_h, original_w = original_sizes[i]
            scale = scales[i]
            
            for box in detections:
                try:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    
                    # Adjust coordinates back to original scale
                    if scale < 1:  # Only rescale if we actually resized the image
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
                except Exception as box_error:
                    logger.error(f"Error processing detection box: {str(box_error)}")
                    continue
                    
            idx = valid_indices[i]
            filename = os.path.basename(filepaths[i])
            
            batch_results.append({
                "filename": filename,
                "predictions": predictions,
                "image_info": {
                    "original_size": [original_w, original_h],
                    "processed_size": [image_batch[i].shape[1], image_batch[i].shape[0]]
                }
            })
        
        # Clean up temporary files
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        return jsonify({
            "results": batch_results,
            "timing": {
                "preprocessing": preprocess_time,
                "inference": inference_time,
                "total": total_time
            },
            "batch_size": len(image_batch)
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error in batch processing: {str(e)}")
        logger.error(error_details)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/class_names', methods=['GET'])
def get_class_names():
    """Return the class names used by the model"""
    model = get_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if hasattr(model, 'names'):
        return jsonify({
            "class_names": model.names
        })
    else:
        return jsonify({"error": "Class names not available"}), 404

def get_memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mb = process.memory_info().rss / 1024 / 1024
        return f"{mb:.2f} MB"
    except ImportError:
        return "psutil not available"

def log_memory_usage(tag=""):
    """Log current memory usage"""
    memory = get_memory_usage()
    logger.info(f"Memory usage {tag}: {memory}")

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

    # Preload the model in a separate thread to avoid cold start
    import threading
    threading.Thread(target=get_model).start()
    
    # Use gunicorn in production
    if os.environ.get('IS_PRODUCTION', 'false').lower() == 'true':
        # Let gunicorn handle the app
        logger.info("Running in production mode with gunicorn")
    else:
        # For development, use the Flask dev server
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting development server on port {port}")
        app.run(debug=False, host='0.0.0.0', port=port, threaded=True)