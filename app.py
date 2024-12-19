from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('median_values.pkl', 'rb') as f:
    median_values = pickle.load(f)

# Database connection parameters
HOST = 'branchhomeworkdb.cv8nj4hg6yra.ap-south-1.rds.amazonaws.com'
PORT = '5432'
USER = 'datascientist'
PASSWORD = os.getenv('DB_PASSWORD')  
DATABASE = 'branchdsprojectgps'

engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Invalid Content-Type. Expected application/json'}), 400

    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({'error': 'Missing user_id in request'}), 400

    user_id = data['user_id']

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({'error': 'Invalid user_id. Must be an integer.'}), 400

    query = text("""
    SELECT u.age, u.cash_incoming_30days, g.accuracy_mean, g.accuracy_std
    FROM user_attributes u
    LEFT JOIN (
        SELECT user_id,
               AVG(accuracy) as accuracy_mean,
               STDDEV(accuracy) as accuracy_std
        FROM gps_fixes
        GROUP BY user_id
    ) g ON u.user_id = g.user_id
    WHERE u.user_id = :user_id
    """)

    try:
        user_data = pd.read_sql(query, engine, params={'user_id': user_id})
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

    if user_data.empty:
        return jsonify({'error': f'User with user_id {user_id} not found'}), 404

    # Preprocess
    numerical_cols = ['age', 'cash_incoming_30days', 'accuracy_mean', 'accuracy_std']
    user_data = user_data[numerical_cols]
    user_data.fillna(median_values, inplace=True)
    features = scaler.transform(user_data)

    # Prediction
    proba = model.predict_proba(features)[0][1]
    outcome = 'Approved' if proba >= 0.5 else 'Declined'

    return jsonify({
        'user_id': user_id,
        'probability': float(proba),
        'outcome': outcome
    })

if __name__ == '__main__':
    app.run(debug=True)
