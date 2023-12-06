import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if round(prediction[0], 2) == 0:
        return render_template('index.html', prediction_text='No heart disease. You are Fine.')
    else:
        return render_template('index.html', prediction_text='Yes, you have chances of heart disease.')


if __name__ == "__main__":
    app.run(debug=True)