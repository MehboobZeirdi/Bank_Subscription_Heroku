from flask import Flask, request, url_for, redirect, render_template, jsonify
# from flask_cors import CORS,cross_origin
# from pycaret.regression import *
# from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np
import xgboost

# print(xgboost.__version__)

app = Flask(__name__)

cols = ['age', 'job', 'marital', 'default', 'housing', 'loan', 'contact',
       'month', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
       'day_of_week_ordinal', 'education_ordinal']


# Loading the saved models into memory

loaded_model = pickle.load(open('xgboost_model.pickle', 'rb'))




@app.route('/')
def home():
    return render_template("home.html")



@app.route('/predict',methods=['POST', 'GET']) # route to show the predictions in a web UI
# @cross_origin()
def predict():

    int_feature = [float(x) for x in request.form.values()]
    final = [np.array(int_feature)]
    predicted = loaded_model.predict(final)
    a = "Subcscribe"
    b = "Not Subscribe"
    if predicted == 0:
        return render_template('home.html', pred='Person will {}'.format(b))
    else:
        # return a
        return render_template('home.html', pred='Person will {}'.format(a))



@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = loaded_model.predict(data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


# if __name__ == '__main__':
#     app.run(debug=True)


