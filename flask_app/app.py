import pandas as pd
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import iop
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'model', 'Healthy_or_Sick.h5')
model = load_model(model_path, compile=False)

def preprocess_image(image_path):
    image = Image.open(image_path).resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def default():
    return render_template('index.html')

@app.route("/home")
def home():
    return render_template('home.html') 

@app.route("/overview")
def overview():
    return render_template('overview.html') 

@app.route("/knn", methods=["GET", "POST"])
@app.route('/knn/<imagesrc>', methods=["GET", "POST"])
def knn_route():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if file:
            filename = file.filename
            filepath = os.path.join('flask_app/static/images/Leaf', filename)
            #os.makedirs('uploads', exist_ok=True)
            file.save(filepath)

            image = preprocess_image(filepath)
            prediction = model.predict(image)
            rslt = (np.round(prediction[0]).astype(int))

            if rslt == 0:
                return render_template('knn_healthy.html', imagesrc=filename)
            else:
                return render_template('knn_unhealthy.html', imagesrc=filename)
    else:
        return render_template('knn.html')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

@app.route("/uploads", methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        image = preprocess_image(filepath)
        prediction = model.predict(image)
        rslt = (np.round(prediction[0]).astype(int))

        if rslt == 0:
            return jsonify({"result": "Healthy"})
        else:
            return jsonify({"result": "Sick"})

if __name__ == '__main__':
    app.run(debug=True, port=5010)