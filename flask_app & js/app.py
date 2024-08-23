from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route("/")
def default():
    return render_template('index.html')

@app.route("/home")
def home():
    return render_template('home.html') 

@app.route("/overview")
def overview():
    return render_template('overview.html') 

@app.route("/knn")
def knn_route():
    return render_template('knn.html')

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html') 

if __name__ == '__main__':
    app.run(debug=True, port=5010)