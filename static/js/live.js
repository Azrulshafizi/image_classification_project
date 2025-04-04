document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const videoElement = document.getElementById('videoElement');
    const canvasElement = document.getElementById('canvasElement');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const resultsElement = document.getElementById('resultsElement');
    const fpsCounter = document.getElementById('fpsCounter');
    const objectsCounter = document.getElementById('objectsCounter');
    
    // Canvas context
    const canvasContext = canvasElement.getContext('2d');
    
    // Detection settings
    const detectionInterval = 100; // milliseconds between detections (adjust for performance)
    let isDetecting = false;
    let detectionTimer = null;
    
    // FPS calculation
    let frameCount = 0;
    let lastFrameTime = performance.now();
    let fps = 0;
    
    // Video stream
    let stream = null;
    
    // COCO class names for YOLOv8
    const cocoClasses = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ];
    
    // Color palette for different object classes
    const colors = [
        '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86',
        '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF',
        '#520085', '#CB38FF', '#FF95C8', '#FF37C7'
    ];
    
    // Start detection
    startButton.addEventListener('click', async function() {
        if (isDetecting) return;
        
        try {
            // First, initialize the camera if not already
            if (!stream) {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        facingMode: 'environment',
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    }, 
                    audio: false 
                });
                videoElement.srcObject = stream;
                
                // Wait for video metadata to load to set canvas dimensions
                await new Promise(resolve => {
                    videoElement.onloadedmetadata = () => {
                        canvasElement.width = videoElement.videoWidth;
                        canvasElement.height = videoElement.videoHeight;
                        
                        // Hide video element as we'll display frames on the canvas
                        videoElement.style.display = 'none';
                        canvasElement.style.display = 'block';
                        resolve();
                    };
                });
            }
            
            // Update UI
            startButton.disabled = true;
            stopButton.disabled = false;
            resultsElement.innerHTML = '<p>Starting real-time detection...</p>';
            
            // Start detection loop
            isDetecting = true;
            detectObjects();
            
            // Start rendering the camera feed to canvas
            renderFrame();
            
        } catch (error) {
            console.error('Error starting detection:', error);
            resultsElement.innerHTML = `<p class="error">Error starting camera: ${error.message}</p>`;
        }
    });
    
    // Stop detection
    stopButton.addEventListener('click', function() {
        stopDetection();
    });
    
    // Stop detection and cleanup
    function stopDetection() {
        isDetecting = false;
        
        if (detectionTimer) {
            clearTimeout(detectionTimer);
            detectionTimer = null;
        }
        
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
            stream = null;
        }
        
        // Reset UI
        startButton.disabled = false;
        stopButton.disabled = true;
        fpsCounter.textContent = 'FPS: --';
        objectsCounter.textContent = 'Objects: --';
        
        // Clear canvas
        canvasContext.clearRect(0, 0, canvasElement.width, canvasElement.height);
        resultsElement.innerHTML = '<p>Detection stopped</p>';
    }
    
    // Render video frame to canvas
    function renderFrame() {
        if (!isDetecting) return;
        
        if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
            // Draw video frame to canvas
            canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            
            // Calculate FPS
            frameCount++;
            const now = performance.now();
            const elapsed = now - lastFrameTime;
            
            if (elapsed > 1000) { // Update FPS every second
                fps = Math.round((frameCount * 1000) / elapsed);
                frameCount = 0;
                lastFrameTime = now;
                fpsCounter.textContent = `FPS: ${fps}`;
            }
        }
        
        requestAnimationFrame(renderFrame);
    }
    
    // Object detection loop
    async function detectObjects() {
        if (!isDetecting) return;
        
        try {
            // Capture current frame from video
            canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            
            // Convert canvas to blob for server
            const blob = await new Promise(resolve => {
                canvasElement.toBlob(resolve, 'image/jpeg', 0.8);
            });
            
            // Create form data
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            
            // Send to server for detection
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                console.error('Detection error:', data.error);
                resultsElement.innerHTML = `<p class="error">Error: ${data.error}</p>`;
            } else {
                // Process and display results
                const predictions = data.predictions || [];
                
                // Draw detections on canvas
                drawDetections(predictions);
                
                // Update object counter
                objectsCounter.textContent = `Objects: ${predictions.length}`;
                
                // Update results list (limit to top 10 for performance)
                if (predictions.length > 0) {
                    // Sort by confidence
                    predictions.sort((a, b) => b.confidence - a.confidence);
                    
                    // Display top results
                    const topPredictions = predictions.slice(0, 10);
                    let resultsHTML = '<ul class="detection-list">';
                    topPredictions.forEach(pred => {
                        const className = cocoClasses[pred.class] || `Class ${pred.class}`;
                        const confidence = Math.round(pred.confidence * 100);
                        resultsHTML += `<li>${className} (${confidence}%)</li>`;
                    });
                    
                    if (predictions.length > 10) {
                        resultsHTML += `<li>+ ${predictions.length - 10} more...</li>`;
                    }
                    
                    resultsHTML += '</ul>';
                    resultsElement.innerHTML = resultsHTML;
                } else {
                    resultsElement.innerHTML = '<p>No objects detected</p>';
                }
            }
        } catch (error) {
            console.error('Error during detection cycle:', error);
        }
        
        // Schedule next detection (if still detecting)
        if (isDetecting) {
            detectionTimer = setTimeout(detectObjects, detectionInterval);
        }
    }
    
    // Draw bounding boxes and labels on the canvas
    function drawDetections(predictions) {
        // First draw the video frame
        canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Then draw each detection
        predictions.forEach(pred => {
            const [x1, y1, x2, y2] = pred.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            const className = cocoClasses[pred.class] || `Class ${pred.class}`;
            const confidence = Math.round(pred.confidence * 100);
            
            // Select color based on class (modulo to handle more classes than colors)
            const color = colors[pred.class % colors.length];
            
            // Draw rectangle
            canvasContext.strokeStyle = color;
            canvasContext.lineWidth = 3;
            canvasContext.strokeRect(x1, y1, width, height);
            
            // Draw background for text
            canvasContext.fillStyle = color;
            const textMetrics = canvasContext.measureText(`${className} ${confidence}%`);
            const textWidth = textMetrics.width;
            canvasContext.fillRect(x1, y1 - 25, textWidth + 10, 25);
            
            // Draw text
            canvasContext.fillStyle = '#FFFFFF';
            canvasContext.font = 'bold 16px Arial';
            canvasContext.fillText(`${className} ${confidence}%`, x1 + 5, y1 - 7);
        });
    }
    
    // Clean up when leaving page
    window.addEventListener('beforeunload', function() {
        stopDetection();
    });
});