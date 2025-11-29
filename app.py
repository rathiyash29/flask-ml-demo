# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# ----------------------------------------
# Load the trained model
# ----------------------------------------
model_path = 'model.pkl'

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ----------------------------------------
# Initialize Flask app
# ----------------------------------------
app = Flask(__name__)

# ----------------------------------------
# Home Page Route
# ----------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ----------------------------------------
# Prediction Route
# ----------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]

        # Make prediction
        prediction = model.predict(final_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'

        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# ----------------------------------------
# Run the app locally only
# ----------------------------------------
if __name__ == "__main__":
    # debug=False so it behaves same as Render
    app.run(host="0.0.0.0", port=5000, debug=False)
