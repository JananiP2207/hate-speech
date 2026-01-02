from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================
# Config
# =====================
MODEL_PATH = "hate_speech_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100

# Load model
model = load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

# Flask App
app = Flask(__name__)

def predict_text(text, threshold=0.5):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = float(model.predict(padded)[0][0])
    label = "Hate Speech" if prob >= threshold else "Not Hate Speech"
    return {"label": label, "confidence": round(prob, 2)}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("message", "")
    result = predict_text(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
