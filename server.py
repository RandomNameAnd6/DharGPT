from flask import Flask, request, jsonify, render_template
from app import generate_text_with_typing

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_text", methods=["POST"])
def generate_text():
    data = request.get_json()
    input_text = data.get('input_text', '')
    max_length = int(data.get('max_length', 100))
    temperature = float(data.get('temperature', 1))
    top_k = int(data.get('top_k', 50))
    top_p = float(data.get('top_p', 0.9))
    repetition_penalty = float(data.get('repetition_penalty', 1.0))

    generated_text = list(generate_text_with_typing(input_text, max_length, temperature, top_k, top_p, repetition_penalty))
    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)