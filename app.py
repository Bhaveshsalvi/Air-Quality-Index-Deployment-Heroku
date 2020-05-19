# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:37:35 2020

@author: Bhavesh
"""


from flask import Flask,render_template,url_for, request,jsonify

import pandas as pd
import numpy as np
 
import pickle

loaded_model = pickle.load(open("Random_forest_model.pkl","rb" ))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_feat = [np.array(int_features)]
    pred = loaded_model.predict(final_feat)
    print(pred[0])
                                 
    return render_template('home.html',prediction_text = "AQI for Bangalore : {}".format(pred[0]))


@app.route('/predict_API',methods=['POST'])
def predict_API():
    
    data = request.get._json(force=True)
    prediction = loaded_model.predict([np.array(list(data.value()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
