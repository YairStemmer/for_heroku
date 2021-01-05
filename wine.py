from flask import Flask, request, jsonify, make_response
import pickle
import numpy as np

app = Flask(__name__)

clf = pickle.load(open("regressor_rf.pkl", "rb"))

@app.route('/')
def welcome():
    return 'Predict wine quality with following parameters:\n' \
           'acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol'

@app.route('/predict')
def predict():
    ph = request.args.get('pH')
    fixed_acidity = request.args.get('fixed acidity')
    volatile_acidity = request.args.get('volatile acidity')
    sulphates = request.args.get('sulphates')
    chlorides = request.args.get('chlorides')
    alcohol = request.args.get('alcohol')
    residual_sugar = request.args.get('residual sugar')
    citric_acid = request.args.get('citric acid')
    free_sulfur_dioxide = request.args.get('free sulfur dioxide')
    total_sulfur_dioxide = request.args.get('total sulfur dioxide')
    density = request.args.get('density')

    parameters = [ph, fixed_acidity, volatile_acidity, sulphates, chlorides, alcohol, residual_sugar, citric_acid,
                  free_sulfur_dioxide, total_sulfur_dioxide, density]
    print(np.array(parameters).astype(float))

    prediction = str(clf.predict(np.array(parameters).reshape(1, -1))[0])
    print(prediction)
    return prediction


@app.route("/json", methods=["POST"])
def json_predict():
    if request.is_json:
        req = request.get_json()['to_predict']
        predictions = []
        for parameters in req:
            predictions.append(clf.predict(np.array(parameters).reshape(1, -1))[0])

        response_body = {
            "predictions": predictions
        }
        res = make_response(jsonify(response_body), 200)
        return res

    else:
        return make_response(jsonify({"message": "Request body must be JSON"}), 400)


if __name__ == '__main__':
    app.run()