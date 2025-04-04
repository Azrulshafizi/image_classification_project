
        // COCO class names (80 classes) - matches YOLOv8n default model
        const classNames = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ];
        
        // DOM Elements
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        const dropArea = document.getElementById('drop-area');
        const fileInput = document.getElementById('file-input');
        const uploadButton = document.getElementById('upload-button');
        const preview = document.getElementById('preview');
        const imageContainer = document.getElementById('image-container');
        const results = document.getElementById('results');
        const detectionInfo = document.getElementById('detection-info');
        const loading = document.getElementById('loading');
        const detectObjectsBtn = document.getElementById('detectObjects');
        const clearResultsBtn = document.getElementById('clearResults');
        
        // Camera elements
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const startCameraBtn = document.getElementById('startCamera');
        const captureImageBtn = document.getElementById('captureImage');
        const switchCameraBtn = document.getElementById('switchCamera');
        
        let stream = null;
        let currentFacingMode = 'environment'; // 'environment' for back camera, 'user' for front camera
        let imageSource = null; // 'camera' or 'upload'
        
        // Tab switching
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Update active tab
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Show corresponding content
                const tabId = tab.getAttribute('data-tab');
                tabContents.forEach(content => content.classList.remove('active'));
                document.getElementById(`${tabId}-content`).classList.add('active');
                
                // Reset state when switching tabs
                clearResults();
                resetControls();
                
                // Stop camera if switching away from camera tab
                if (tabId !== 'camera' && stream) {
                    stopCamera();
                }
            });
        });
        
        // Event listeners for upload
        uploadButton.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);
        
        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.classList.add('highlight');
        });
        
        dropArea.addEventListener('dragleave', () => {
            dropArea.classList.remove('highlight');
        });
        
        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.classList.remove('highlight');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });
        
        // Event listeners for camera
        startCameraBtn.addEventListener('click', toggleCamera);
        captureImageBtn.addEventListener('click', captureImage);
        switchCameraBtn.addEventListener('click', switchCamera);
        
        // Common controls
        detectObjectsBtn.addEventListener('click', detectObjects);
        clearResultsBtn.addEventListener('click', clearResults);
        
        // Handle file selection
        function handleFileSelect() {
            const file = fileInput.files[0];
            if (!file) return;
            
            imageSource = 'upload';
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
                clearDetections();
                results.style.display = 'none';
                detectObjectsBtn.disabled = false;
                clearResultsBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        // Camera functions
        async function toggleCamera() {
            if (stream) {
                stopCamera();
                startCameraBtn.textContent = 'Start Camera';
            } else {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { 
                            facingMode: currentFacingMode,
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        }
                    });
                    video.srcObject = stream;
                    startCameraBtn.textContent = 'Stop Camera';
                    captureImageBtn.disabled = false;
                    switchCameraBtn.disabled = false;
                    
                    // Hide preview if showing
                    preview.style.display = 'none';
                    results.style.display = 'none';
                    clearDetections();
                    
                } catch (err) {
                    console.error('Error accessing camera:', err);
                    alert(`Cannot access camera: ${err.message}`);
                }
            }
        }
        
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                video.srcObject = null;
                captureImageBtn.disabled = true;
                switchCameraBtn.disabled = true;
            }
        }
        
        function captureImage() {
            if (!stream) return;
            
            imageSource = 'camera';
            
            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Draw video frame to canvas
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to image and display
            preview.src = canvas.toDataURL('image/jpeg');
            preview.style.display = 'block';
            
            // Enable detection
            detectObjectsBtn.disabled = false;
            clearResultsBtn.disabled = false;
        }
        
        async function switchCamera() {
            // Switch facing mode
            currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
            
            // Stop current stream
            if (stream) {
                stopCamera();
            }
            
            // Restart with new facing mode
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        facingMode: currentFacingMode,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });
                video.srcObject = stream;
                captureImageBtn.disabled = false;
                switchCameraBtn.disabled = false;
            } catch (err) {
                console.error('Error switching camera:', err);
                alert(`Cannot switch camera: ${err.message}`);
            }
        }
        
        // Object detection
        function detectObjects() {
            if (!preview.src) return;
            
            loading.style.display = 'block';
            clearDetections();
            
            if (imageSource === 'upload') {
                const file = fileInput.files[0];
                if (!file) return;
                
                uploadImage(file);
            } else if (imageSource === 'camera') {
                // Convert canvas to blob and upload
                canvas.toBlob(blob => {
                    uploadImage(blob);
                }, 'image/jpeg', 0.95);
            }
        }
        
        
function uploadImage(fileOrBlob) {
    const formData = new FormData();
    
    // Add file with proper name and type information
    if (fileOrBlob instanceof Blob && !(fileOrBlob instanceof File)) {
        // If it's a blob from canvas, give it a proper name and type
        const capturedFile = new File([fileOrBlob], "captured-image.jpg", {
            type: "image/jpeg"
        });
        formData.append('image', capturedFile);
        console.log("Uploading captured image as File object");
    } else {
        // If it's already a File object
        formData.append('image', fileOrBlob);
        console.log(`Uploading file: ${fileOrBlob.name}, size: ${fileOrBlob.size} bytes, type: ${fileOrBlob.type}`);
    }
    
    // Show loading indicator
    loading.style.display = 'block';
    clearResultsBtn.disabled = true;
    detectObjectsBtn.disabled = true;
    
    // Set a timeout to detect if the request is taking too long
    const timeoutId = setTimeout(() => {
        alert('The request is taking longer than expected. The server might be busy or unavailable.');
    }, 15000); // 15 seconds timeout
    
    // Log the request being sent
    console.log("Sending request to server...");
    
    fetch('/predict', {
        method: 'POST',
        body: formData,
        mode: 'cors',
        cache: 'no-cache',
        credentials: 'same-origin',
        redirect: 'follow',
        referrerPolicy: 'no-referrer',
    })
    .then(response => {
        clearTimeout(timeoutId); // Clear the timeout
        console.log(`Server responded with status: ${response.status}`);
        
        // Try to parse the response as JSON
        return response.json().then(data => {
            // If we successfully got JSON, return it with the status
            return { data, status: response.status, ok: response.ok };
        }).catch(err => {
            // If JSON parsing failed, create an error object
            return { 
                data: { error: "Failed to parse response" }, 
                status: response.status, 
                ok: false 
            };
        });
    })
    .then(({ data, status, ok }) => {
        if (!ok) {
            // Handle error response
            console.error('Error response:', data);
            throw new Error(data.error || `Server error: ${status}`);
        }
        
        // Handle successful response
        console.log('Success response:', data);
        displayResults(data);
    })
    .catch(error => {
        console.error('Request failed:', error);
        
        // Handle specific error cases
        if (error.message.includes('Failed to fetch')) {
            alert('Network error: Could not connect to the server. Please check your internet connection and try again.');
        } else if (error.message.includes('BAD REQUEST')) {
            alert('The server rejected the image. It may be too large, corrupted, or in an unsupported format. Please try a different image.');
        } else {
            alert('Error processing image: ' + error.message);
        }
    })
    .finally(() => {
        // Always hide loading indicator and re-enable buttons
        loading.style.display = 'none';
        clearResultsBtn.disabled = false;
        detectObjectsBtn.disabled = false;
    });
}

// Add this function to validate images before upload
function validateImage(fileOrBlob) {
    return new Promise((resolve, reject) => {
        // If it's a blob from canvas capture, it should be valid
        if (fileOrBlob instanceof Blob && !(fileOrBlob instanceof File)) {
            resolve(true);
            return;
        }
        
        // For uploaded files, check size and format
        if (fileOrBlob.size === 0) {
            reject(new Error("File is empty"));
            return;
        }
        
        if (fileOrBlob.size > 16 * 1024 * 1024) {
            reject(new Error("File is too large (max 16MB)"));
            return;
        }
        
        // Check if it's a valid image by loading it into an Image object
        const img = new Image();
        img.onload = () => resolve(true);
        img.onerror = () => reject(new Error("Invalid image file"));
        
        const url = URL.createObjectURL(fileOrBlob);
        img.src = url;
    });
}

// Modify the detectObjects function to validate before upload
function detectObjects() {
    if (!preview.src) return;
    
    let fileOrBlob;
    
    if (imageSource === 'upload') {
        fileOrBlob = fileInput.files[0];
        if (!fileOrBlob) return;
        
        validateImage(fileOrBlob)
            .then(() => uploadImage(fileOrBlob))
            .catch(error => {
                alert(`Image validation failed: ${error.message}`);
            });
    } else if (imageSource === 'camera') {
        canvas.toBlob(blob => {
            validateImage(blob)
                .then(() => uploadImage(blob))
                .catch(error => {
                    alert(`Camera image validation failed: ${error.message}`);
                });
        }, 'image/jpeg', 0.95);
    }
}
        
        function displayResults(data) {
            clearDetections();
            
            if (!data.predictions || data.predictions.length === 0) {
                detectionInfo.innerHTML = '<p>No objects detected.</p>';
                results.style.display = 'block';
                return;
            }
            
            // Get actual image dimensions
            const imgWidth = preview.naturalWidth;
            const imgHeight = preview.naturalHeight;
            
            // Get displayed image dimensions
            const displayWidth = preview.clientWidth;
            const displayHeight = preview.clientHeight;
            
            // Calculate scale
            const scaleX = displayWidth / imgWidth;
            const scaleY = displayHeight / imgHeight;
            
            // Display detections
            data.predictions.forEach(pred => {
                // Scale bounding box to match displayed image size
                const x1 = pred.bbox[0] * scaleX;
                const y1 = pred.bbox[1] * scaleY;
                const x2 = pred.bbox[2] * scaleX;
                const y2 = pred.bbox[3] * scaleY;
                
                const width = x2 - x1;
                const height = y2 - y1;
                
                // Create box
                const box = document.createElement('div');
                box.className = 'detection-box';
                box.style.left = `${x1}px`;
                box.style.top = `${y1}px`;
                box.style.width = `${width}px`;
                box.style.height = `${height}px`;
                
                // Create label
                const label = document.createElement('div');
                label.className = 'detection-label';
                const className = classNames[pred.class] || `Class ${pred.class}`;
                label.textContent = `${className} ${Math.round(pred.confidence * 100)}%`;
                
                box.appendChild(label);
                imageContainer.appendChild(box);
            });
            
            // Create summary text
            const counts = {};
            data.predictions.forEach(pred => {
                const className = classNames[pred.class] || `Class ${pred.class}`;
                counts[className] = (counts[className] || 0) + 1;
            });
            
            let summaryHTML = '<p>Objects detected:</p><ul>';
            Object.entries(counts).forEach(([className, count]) => {
                summaryHTML += `<li>${className}: ${count}</li>`;
            });
            summaryHTML += '</ul>';
            
            detectionInfo.innerHTML = summaryHTML;
            results.style.display = 'block';
        }
        
        function clearDetections() {
            const boxes = document.querySelectorAll('.detection-box');
            boxes.forEach(box => box.remove());
            results.style.display = 'none';
        }
        
        function clearResults() {
            clearDetections();
            preview.style.display = 'none';
            preview.src = '';
            results.style.display = 'none';
            detectObjectsBtn.disabled = true;
            clearResultsBtn.disabled = true;
            fileInput.value = '';
        }
        
        function resetControls() {
            detectObjectsBtn.disabled = true;
            clearResultsBtn.disabled = true;
        }
        
        // Check camera support on load
        document.addEventListener('DOMContentLoaded', () => {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                document.querySelector('[data-tab="camera"]').style.display = 'none';
                alert('Your browser does not support camera access. The camera tab has been disabled.');
            }
        });
    