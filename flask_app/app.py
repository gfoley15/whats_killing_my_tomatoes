import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)
model_path_Healthy_or_Sick = os.path.join(os.path.dirname(__file__), 'model', 'all_cat.h5')
Healthy_or_Sick = load_model(model_path_Healthy_or_Sick, compile=False)

def preprocess_image(image_path):
    image = Image.open(image_path).resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def default():
    return render_template('home.html')

@app.route("/home")
def home():
    return render_template('home.html') 

@app.route("/overview")
def overview():
    return render_template('overview.html') 

@app.route("/cnn", methods=["GET", "POST"])
@app.route('/cnn/<imagesrc>/<regvalue>', methods=["GET", "POST"])
def cnn_route():
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

            names = {0:"Bacterial spot", 
                     1:"Early blight",
                     2:"healthy",
                     3:"Late blight",
                     4:"Leaf Mold",
                     5:"Mosaic virus",
                     6:"Septoria leaf spot",
                     7:"Target Spot",
                     8:"Two spotted spider mite",
                     9:"Yellow Leaf Curl Virus"}

            predict_x=Healthy_or_Sick.predict(image) 
            classes_x=np.argmax(predict_x,axis=1)
            model_class = names.get(classes_x[0])
            if model_class == "healthy":
                return render_template('cnn_healthy.html', imagesrc=filename, regvalue=model_class)
            else:
                return render_template('cnn_unhealthy.html', imagesrc=filename, regvalue=model_class)

    else:
        return render_template('cnn.html')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

@app.route("/dashboard/Garrett")
def dashboard_garrett():
    return render_template('logistic_regression_model_GF.html')

@app.route("/dashboard/Mohamed")
def dashboard_mohamed():
    return render_template('nn_accuracy_model_MI.html')

@app.route("/dashboard/Amanuel")
def dashboard_amanuel():
    return render_template('cnn_accuracy_model_AM.html')

@app.route("/dashboard/John")
def dashboard_john():
    return render_template('Healthy_Sick_Anlysis_JT.html')

if __name__ == '__main__':
    app.run(debug=True, port=5010)