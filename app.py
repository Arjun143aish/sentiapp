import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template

app = Flask(__name__)

CV = pickle.load(open('CV.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        Vector = CV.transform(data).toarray()
        my_prediction = model.predict(Vector)
        return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
    
