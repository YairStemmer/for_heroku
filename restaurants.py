from flask import Flask, request, jsonify, make_response
import numpy as np
import datetime
from joblib import load

app = Flask(__name__)

clf = load(open("rf_random.joblib", "rb"))


risk_dict = {'Risk 1 (High)':1, 'Risk 2 (Medium)':2, 'Risk 3 (Low)':3}

facility_dict = {
 'Restaurant': 0,
 'TAVERN': 1,
 'Grocery Store': 2,
 'Other': 3,
 'tavern': 4,
 'School': 5,
 'Bakery': 6,
 'Catering': 7,
 'Daycare (2 - 6 Years)': 8,
 'Daycare Above and Under 2 Years': 9,
 "Children's Services Facility": 10,
 'Daycare Combo 1586': 11,
 'PALETERIA': 12,
 'STORE': 13,
 'Mobile Food Preparer': 14,
 'JUICE BAR/GROCERY': 15}

inspection_dict = {
 'Canvass': 0,
 'Complaint': 1,
 'Canvass Re-Inspection': 2,
 'Short Form Complaint': 3,
 'Complaint Re-Inspection': 4,
 'Other': 5,
 'Suspected Food Poisoning': 6,
 'Consultation': 7,
 'Suspected Food Poisoning Re-inspection': 8,
 'Recent Inspection': 9,
 'License Re-Inspection': 10,
 'License': 11}



@app.route('/')
def welcome():
    return 'This app will tell you the future of the next restaurant inspection. Good luck and clean your kitchen!!'


@app.route("/json", methods=["POST"])
def json_predict():
    if request.is_json:
        req = request.get_json()
        params = {}

        # Facility, Risk, Inspection type, Violations, Results, inspection time
        # Facility_enc, Risk_enc, Inspection_enc, Violations_enc, Results_enc, time_delta
        #predictions = []

        params['Facility_enc'] = int(facility_dict[req['Facility_enc']])
        params['Risk_enc'] = int(req['Risk_enc'])
        params['Inspection_enc'] = int(inspection_dict[req['Inspection_enc']])
        params['Violations_enc'] = len(req['Violations_enc'])

        params['Results_enc'] = int(req['Results_enc'])
        inspection_date = req['Inspection_Date']
        d0 = datetime.datetime.strptime(inspection_date, '%Y-%m-%d')
        d1 = datetime.datetime.now()
        delta = (d1 - d0).days
        params['time_delta'] = delta

        parameters_array = [params['Facility_enc'], params['Risk_enc'], params['Inspection_enc'],
                            params['Violations_enc'], params['Results_enc'], params['time_delta']]

        prediction = clf.predict(np.array(parameters_array).reshape(1, -1))[0]

        response_body = {
            "License": int(req['License']),
            "Result_pred": int(prediction)
        }
        print(response_body)

        res = make_response(jsonify(response_body), 200)
        return res

    else:
        return make_response(jsonify({"message": "Request body must be JSON"}), 400)


if __name__ == '__main__':
    app.run()
