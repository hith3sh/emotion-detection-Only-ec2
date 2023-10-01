from flask import Flask, render_template, send_from_directory, request
import numpy as np
from predict import detect_emotions
import os
from PIL import Image
import cv2
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Handle file upload
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Save the uploaded file
        file_path = os.path.join('static', 'uploads', uploaded_file.filename)
        uploaded_file.save(file_path)

        try:
            # Predict emotion and add bounding box
            img = cv2.imread(file_path)
            emotion_img = detect_emotions(img)
            output_path = os.path.join('static', 'output', uploaded_file.filename)
            emotion_img = cv2.cvtColor(emotion_img, cv2.COLOR_BGR2RGB)
            emotion_img = Image.fromarray(emotion_img)
            emotion_img.save(output_path)
            # Log success message
            logging.info(f"File {uploaded_file.filename} processed successfully")

        except Exception as e:
            # Log error message
            logging.error(f"An error occurred while processing file {uploaded_file.filename}: {e}")
            
        # Pass the filename of the processed image to the index.html file
        return render_template('index.html', image=uploaded_file.filename)

    # else
    return 'Error uploading file'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
