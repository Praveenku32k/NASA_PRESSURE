# Important library.
"""
Author: Praveen Kumar
Company: TuringMinds.ai
Date   : 25th nov 2022
Project : NASA DATA  
"""

import pickle
import flask
import numpy as np
import pandas as pd
from flask import Flask, request, app,jsonify
from flask import Response
from flask_cors import CORS

## Start the Flask name 
app=Flask(__name__)

# Loading the pickle file.
# model=pickle.load(open('model.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict_api',methods=['POST'])

def predict_api():

    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)




## Point of exection will start from here.
if __name__=="__main__":
    app.run(debug=True)
