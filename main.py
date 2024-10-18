from flask import Flask, request, jsonify, send_file
import pandas as pd
import pickle
import io

app = Flask(__name__)

# Load the scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("anomaly_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return '''
        <h1>Anomaly Detection API</h1>
        <p>Use the following URL to upload your CSV file:</p>
        <code>http://localhost:5000/predict?csv_path=your_file_path</code>
    '''

@app.route("/predict", methods=["GET"])
def predict():
    try:
        csv_path = request.args.get("csv_path")
        if not csv_path:
            return jsonify({"error": "Please provide a valid CSV file path as a query parameter."}), 400

        # Load the CSV file from the provided path
        data = pd.read_csv(csv_path)

        # Ensure required columns are present
        required_columns = ['CPU_Usage(%)', 'Memory_Usage(%)', 'Latency(ms)']
        if not all(col in data.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns: {required_columns}"}), 400

        # Create interaction features
        data['CPU_RAM_Interaction'] = data['CPU_Usage(%)'] * data['Memory_Usage(%)']
        data['Latency_per_CPU'] = data['Latency(ms)'] / (data['CPU_Usage(%)'] + 1)

        # Prepare the input data
        input_data = data[['CPU_Usage(%)', 'Memory_Usage(%)', 'Latency(ms)', 
                           'CPU_RAM_Interaction', 'Latency_per_CPU']]
        input_scaled = scaler.transform(input_data)

        # Get predictions from the model
        anomaly_scores = model.decision_function(input_scaled)
        anomaly_status = (anomaly_scores < 0).astype(int)

        # Add results to the DataFrame
        data['Anomaly_Score'] = anomaly_scores
        data['Anomaly_Status'] = anomaly_status

        # Convert results to JSON
        result = data[['CPU_Usage(%)', 'Memory_Usage(%)', 'Latency(ms)', 
                       'Anomaly_Score', 'Anomaly_Status']].to_dict(orient='records')

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
