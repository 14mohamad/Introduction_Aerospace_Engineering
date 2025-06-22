"""
Flask server for star identification
Takes image as input and returns annotated image with identified stars
"""
from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import pandas as pd
import os
import io
import base64
from werkzeug.utils import secure_filename
import tempfile
import uuid

# Import the star identification algorithm
from star_algorithm import (
    CompleteCatalog, 
    identify_and_annotate_stars,

)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize catalog on startup
catalog = CompleteCatalog()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Star Identifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: auto; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; }
            #result { margin-top: 20px; }
            img { max-width: 100%; height: auto; }
            pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }
            .warning { color: #ff6600; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Star Identifier</h1>
            <p>Upload an astronomical image to identify stars</p>
            
            <div class="upload-area">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="fileInput" accept="image/*" required>
                    <br><br>
                    <label>Threshold Value: 
                        <input type="range" id="threshold" min="50" max="200" value="120">
                        <span id="thresholdValue">120</span>
                    </label>
                    <br><br>
                    <button type="submit">Identify Stars</button>
                </form>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('threshold').oninput = function() {
                document.getElementById('thresholdValue').textContent = this.value;
            };
            
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('fileInput');
                const threshold = document.getElementById('threshold').value;
                
                formData.append('image', fileInput.files[0]);
                formData.append('threshold', threshold);
                
                document.getElementById('result').innerHTML = '<p>Processing...</p>';
                
                try {
                    const response = await fetch('/identify', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        let errorHtml = `<p class="warning">Warning: ${data.error}</p>`;
                        if (data.image) {
                            errorHtml += `<img src="${data.image}" alt="Detected stars">`;
                        }
                        document.getElementById('result').innerHTML = errorHtml;
                    } else {
                        let resultHtml = '<h2>Results</h2>';
                        if (data.info.solve_method === 'filename pattern') {
                            resultHtml += '<p class="warning">Note: Coordinates estimated from filename</p>';
                        }
                        resultHtml += `
                            <img src="${data.image}" alt="Identified stars">
                            <h3>Summary</h3>
                            <pre>${JSON.stringify(data.info, null, 2)}</pre>
                        `;
                        document.getElementById('result').innerHTML = resultHtml;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    '''

@app.route('/identify_fast', methods=['POST'])
def identify_fast():
    """Fast endpoint that returns identification data with annotated image URL"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Get parameters
        threshold = int(request.form.get('threshold', 120))
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Use the algorithm module to process the image
        annotated_img, result_info = identify_and_annotate_stars(
            img, 
            filename=file.filename,
            threshold_value=threshold
        )
        
        # Save annotated image
        unique_filename = f"star_result_{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        cv2.imwrite(image_path, annotated_img)
        
        return jsonify({
            'image_url': f"{request.scheme}://{request.host}/uploads/{unique_filename}",
            'info': result_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/identify', methods=['POST'])
def identify():
    """Endpoint to identify stars in uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Get parameters
        threshold = int(request.form.get('threshold', 120))
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process image
        annotated_img, result_info = identify_and_annotate_stars(
            img, 
            filename=file.filename,
            threshold_value=threshold
        )
        
        # Encode result image to base64
        _, buffer = cv2.imencode('.png', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'info': result_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/identify_raw', methods=['POST'])
def identify_raw():
    """Endpoint that returns the annotated image directly (for API usage)"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Get parameters
        threshold = int(request.form.get('threshold', 120))
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process image
        annotated_img, _ = identify_and_annotate_stars(
            img, 
            filename=file.filename,
            threshold_value=threshold
        )
        
        # Return image directly
        _, buffer = cv2.imencode('.png', annotated_img)
        
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/png',
            as_attachment=True,
            download_name=f'identified_{secure_filename(file.filename)}'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Static file serving for uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded image files"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)