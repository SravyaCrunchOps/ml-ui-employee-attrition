import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# sart flask app
app = Flask(__name__)

# Cors
CORS(app)

# load the model and the scaler
# if os.path.isfile("D:/machine learning/basics/my_model_lr.pkl"):
if os.path.isfile("C:/Users/DELL/OneDrive/Desktop/CrunchOps/ml/employee-attrition/my_model_lr.pkl"):
    # with open("D:/machine learning/basics/my_model_lr.pkl", "rb") as f:
    with open("C:/Users/DELL/OneDrive/Desktop/CrunchOps/ml/employee-attrition/my_model_lr.pkl", "rb") as f:
        model_lr, scaler, OrdinalEncoder = pickle.load(f)
else:
    raise FileNotFoundError

# data pre-processing for new data
columns_to_encode = ['Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Education Level', 'Job Level', 'Company Size', 'Company Reputation', 'Employee Recognition']

categories=[
    ['Poor', 'Fair', 'Good', 'Excellent'],
    ['Low', 'Medium', 'High', 'Very High'],
    ['Low', 'Below Average', 'Average', 'High'],
    ["High School", "Bachelor’s Degree", "Master’s Degree", "Associate Degree", "PhD"],
    ['Entry', 'Mid', 'Senior'],
    ['Small', 'Medium', 'Large'],
    ['Poor', 'Fair', 'Good', 'Excellent'],
    ['Low', 'Medium', 'High', 'Very High'],
]

# define numerical encoder... binary encoder
emp_bool_map = ['Overtime', 'Remote Work', 'Opportunities']

# home page
@app.route('/')
def index():
    return render_template('index.html')

# start route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # convert this requested data to pd.dataframe
    X_new = pd.DataFrame([data])

    # apply pre-processing
    oe = OrdinalEncoder(categories=categories)
    X_new[columns_to_encode] = oe.fit_transform(X_new[columns_to_encode]).astype('int')

    for name in emp_bool_map:
        X_new[name] = X_new[name].map({'No': 0, 'Yes': 1})

    # Define the function to map income ranges to ordinal values
    def map_monthly_income(income):
        if 1200 <= income <= 5000:
            return 0
        elif 5001 <= income <= 12000:
            return 1
        elif 12001 <= income <= 20000:
            return 2
        elif 20001 <= income <= 30000:
            return 3
        elif income >= 30001:
            return 4
        else:
            return -1  # Handle any unexpected values

    X_new['Monthly Income'] = X_new['Monthly Income'].apply(map_monthly_income)

    # use standard scaler to scale the features
    features = scaler.transform(X_new)
    
    # predict the output
    y_pred = model_lr.predict(features)

    prediction = "Left" if y_pred[0] == 1 else "Stayed"

    return jsonify({'prediction': prediction})

if __name__=='__main__':
    app.run(debug=True)
