import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pfld import PFLDInference, AuxiliaryNet
from detector import detect_faces
import subprocess
import argparse

import numpy as np
import cv2

import torch
import torchvision


UPLOAD_FOLDER = 'uploads'  # Changed to 'uploads' (without '/')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_file(filename):
    # Add your existing script logic here
    # Make sure to replace input() calls with filename directly

    # Example:
    command = ["python", "your_existing_script.py", "--model_path", "path/to/your/model.pth", "--filename", filename]
    subprocess.run(command, check=True)

@app.route('/media/upload', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        return jsonify({'error': 'media not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        process_file(file_path)  # Call your processing function
        return jsonify({'msg': 'media uploaded successfully'})
    return jsonify({'error': 'file type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
