import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd 

app = Flask(__name__)

## Load the model
model=pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict_api',methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    
    # Extract title from the JSON data
    title = data['data']['title']
    
    # Vectorize the title using the loaded TfidfVectorizer
    title_vectorized = vectorizer.transform([title])
    
    # Make prediction using the pre-trained model
    prediction = model.predict(title_vectorized)[0]
    prediction = int(prediction)
    
    # Return prediction result
    return jsonify({'prediction': prediction})    

if __name__ =="__main__":
    app.run(debug=True)

