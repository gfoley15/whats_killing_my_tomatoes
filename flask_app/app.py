import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
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
@app.route('/knn/<imagesrc>/<regvalue>', methods=["GET", "POST"])
def knn_route():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files['image']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if file:
            filename = file.filename
            filepath = os.path.join('flask_app\static\images\Leaf', filename)
            file.save(filepath)

            image = preprocess_image(filepath)
            prediction = model.predict(image)
            rslt = (np.round(prediction[0]).astype(int))

            valpercent = prediction[0].astype(float)
            failpercent = str(round((valpercent[0]*100), ndigits=2))

            valpercent_inv = 1 - prediction[0].astype(float)
            passpercent = str(round((valpercent_inv[0]*100), ndigits=2))

            if rslt == 0:
                return render_template('knn_healthy.html', imagesrc=filename, regvalue=passpercent)
            else:
                return render_template('knn_unhealthy.html', imagesrc=filename, regvalue=failpercent)
    else:
        return render_template('knn.html')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=5010)