# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:36:28 2020

@author: Anaji
"""
from flask import Flask, render_template,request
from flask_cors import cross_origin

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


from model_3 import FlightFarePredict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        date_dep = request.form["Dep_Time"]
        date_arr = request.form["Arrival_Time"]
        Total_stops = int(request.form["stops"])
        airline = request.form['airline']
        Source = request.form["Source"]
        Dest = request.form["Destination"]
        FlightFarePredictObj = FlightFarePredict()
        my_prediction = FlightFarePredictObj.predictFlightFare(date_dep, 
                                            date_arr, Total_stops, airline,
                                            Source, Dest)
        output=round(my_prediction[0],2)
        return render_template('home.html',prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("home.html")
@app.route('/index')
def index():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
