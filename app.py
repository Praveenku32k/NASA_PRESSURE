# Important library.
import pickle
import flask
from flask import Flask, request, app
from flask import Response
from flask_cors import CORS


app=Flask(__name__)

if __name__=="__main__":
    app.run(debug=True)
