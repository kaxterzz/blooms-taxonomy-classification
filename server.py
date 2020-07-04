from flask import Flask, request, Blueprint, jsonify
import string
import random
from main import predict, train_model, all

app = Flask(__name__)

def randomString(stringLength=5):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

@app.route("/")
def hello():
    return "Hello World from Flask"

@app.route('/all', methods=['POST'])
def full_process():
    try:
        if request.method == 'POST':
            result = all()
            print('result',result)
            return result
        else:
            return False

    except Exception as e:
        print(e)
        return e

@app.route('/train-model', methods=['POST'])
def train():
    try:
        if request.method == 'POST':
            result = train_model()
            print('result',result)
            return result
        else:
            return False

    except Exception as e:
        print(e)
        return e

@app.route('/predict', methods=['POST'])
def predict_taxonomy():
    try:
        if request.method == 'POST':
            f = request.files['file']
            random_file_name = 'received_files/'+randomString()+'.pdf'
            f.save(random_file_name)
            result = predict(random_file_name)
            print('result',result)
            return result
        else:
            return False

    except Exception as e:
        print(e)
        return e

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True,port="3100")