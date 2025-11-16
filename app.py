from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
BASE = os.path.dirname(os.path.abspath(__file__))
ridge_model = pickle.load(open(os.path.join(BASE, 'models/ridge.pkl'), 'rb'))
standard_scaler = pickle.load(open(os.path.join(BASE, 'models/scaler.pkl'), 'rb'))

@app.route("/", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Prepare data
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        scaled = standard_scaler.transform(input_data)

        result = ridge_model.predict(scaled)

        return render_template('home.html', results=float(result[0]))

    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
