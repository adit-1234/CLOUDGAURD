from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the cloudburst prediction model
model = joblib.load("cloudburst_prediction_model.pkl")

# Endpoint to render the HTML page
@app.route('/')
def index():
    return render_template('index3.html')

# Endpoint to handle cloudburst prediction
@app.route('/predict_cloudburst', methods=['POST'])
def predict_cloudburst():
    try:
        # Get input data from the request
        data = request.get_json()

        # Prepare the input data for prediction
        input_data = [[
            data['Intensity (mm/h)'],
            data['Temperature (°C)'],
            data['Humidity (%)'],
            data['Wind Speed (m/s)'],
            data['Wind Direction (°)'],
            data['Atmospheric Pressure (hPa)'],
            data['Precipitation (mm)'],
        ]]

        # Make a prediction using the loaded model
        cloudburst_probability = model.predict_proba(input_data)[:, 1][0]

        # Return the prediction as JSON
        return jsonify({'cloudburst_probability': cloudburst_probability})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
