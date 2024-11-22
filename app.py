from flask import Flask, request, jsonify
from flask_cors import CORS  # Impor Flask-CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)  # This will allow all domains by default

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Load the tokenizer (assuming it was saved as a JSON file)
with open('tokenizer.json', 'r') as handle:
    tokenizer_json = handle.read()  # Read the entire content as a string
    tokenizer = tokenizer_from_json(tokenizer_json)  # Pass the string to tokenizer_from_json

def test_model(texts):
    """
    Test the saved model with new input texts.
    Args:
        texts (list of str): List of text data to classify.
    Returns:
        List of predictions: 0 (Valid) or 1 (Hoax)
    """
    # Tokenize and pad the input
    sequences = tokenizer.texts_to_sequences(texts)  # Convert text to sequences of integers
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')  # Pad the sequences

    # Predict using the model
    predictions = model.predict(padded_sequences)  # Get model's raw predictions
    predicted_labels = [1 if p > 0.5 else 0 for p in predictions]  # Convert probabilities to labels (1 for Hoax, 0 for Valid)

    return predicted_labels

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data JSON dari request
    data = request.get_json()

    # Validasi 'texts' harus ada dan berupa list
    if not data or 'texts' not in data or not isinstance(data['texts'], list):
        return jsonify({"error": "Request harus memiliki field 'texts' yang berupa list string"}), 400

    # Ambil teks dari input
    test_texts = data['texts']

    # Pastikan semua elemen dalam 'texts' adalah string
    if not all(isinstance(text, str) for text in test_texts):
        return jsonify({"error": "'texts' harus berisi elemen string"}), 400

    # Proses prediksi
    results = test_model(test_texts)

    # Format hasil
    response = [{"text": text, "prediction": "Hoax" if label == 1 else "Valid"}
                for text, label in zip(test_texts, results)]

    # Return hasil dalam JSON
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)