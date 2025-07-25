import pickle as pick
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load pre-trained Isolation Forest model and scalers
iso_model = pick.load(open('iso_forest.pkl', 'rb'))
label_encoders = pick.load(open('label_encoders.pkl', 'rb'))
scaler = pick.load(open('scaler.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from request form
        amt = float(request.form['amt'])
        city_pop = int(request.form['city_pop'])
        merchant = request.form['merchant']
        category = request.form['category']
        unix_time = int(request.form['unix_time'])

        # Encode categorical variables using pre-loaded LabelEncoders
        merchant_encoded = label_encoders['merchant'].transform([merchant])[0]
        category_encoded = label_encoders['category'].transform([category])[0]

        # Create a feature array for prediction
        transaction_features = np.array([[
            amt, city_pop, merchant_encoded, category_encoded, unix_time
        ]])

        # Scale features
        transaction_features_scaled = scaler.transform(transaction_features)

        # Make prediction using Isolation Forest model
        anomaly_prediction = iso_model.predict(transaction_features_scaled)

        # Convert the anomaly prediction (-1 for fraud, 1 for normal) into binary fraud label
        is_fraud = 1 if anomaly_prediction[0] == -1 else 0

        # Return the result as JSON
        result = {'is_fraud': is_fraud}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=8080,
        debug=True,
        threaded=True
    )
