import pickle
from flask import Flask, request, jsonify, render_template  # Added render_template here

app = Flask(__name__)

# Load the trained sentiment model pipeline from the pickle file
with open('sentiment_model_pipeline_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data['review']
        prediction = model.predict([review])[0]
        sentiment = "positive" if prediction == 1 else "negative"
        return jsonify({
            'status': 'success',
            'prediction': sentiment
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)