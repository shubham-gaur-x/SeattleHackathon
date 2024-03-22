from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the saved models
logistic_model = load('/Users/shubhamgaur/Desktop/Hackathon/logistic_regression_model.joblib')
knn_model = load('/Users/shubhamgaur/Desktop/Hackathon/knn_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Load test data from request
    data = request.get_json(force=True)
    test_df = pd.DataFrame(data)

    # Make predictions using the logistic regression model
    logistic_predictions = logistic_model.predict(test_df)

    # Make predictions using the K-NN model
    knn_predictions = knn_model.predict(test_df)

    # Prepare response
    response = {
        'logistic_predictions': logistic_predictions.tolist(),
        'knn_predictions': knn_predictions.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
