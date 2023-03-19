
# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-svm-model.pkl'
model = pickle.load(open(filename, 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('main.html')
​
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
​
        # Create a numpy array with the input values
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
​
        # Make a prediction using the machine learning model
        my_prediction = model.predict(data)
​
        # Return the prediction as a response to the request
        return render_template('Result 1.html', prediction=my_prediction)
if __name__ == '__main__':
    app.run(debug=True)
