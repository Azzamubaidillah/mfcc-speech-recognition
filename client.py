from flask import Flask

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    pass
    