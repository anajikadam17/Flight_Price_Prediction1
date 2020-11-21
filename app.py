# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:40:42 2020

@author: Anaji
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/',methods=['GET'])
def home():
    return "YAY!!! its working"


@app.route('/add',methods=['GET'])
def addNum():
    a = 2
    b = 3
    return "The sm of {} and {} is {}".format(a,b,a+b)


if __name__ == "__main__":
    app.run(debug=True)