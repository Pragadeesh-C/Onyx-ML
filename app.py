from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import json
from flask_cors import CORS, cross_origin
# from sklearn.preprocessing import LabelEncoder

app=Flask(__name__)
cors = CORS(app, resources={r"/diet": {"origins": "*"}})
model=pickle.load(open('burn.pkl','rb'))
Diet=pickle.load(open('Diet.pkl','rb'))
Trainer=pickle.load(open('Trainer.pkl','rb'))

app.debug = True

@app.route('/',methods=['GET'])
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    data = request.get_json()
    int_features = [float(data['Gender']),
                    float(data['Age']),
                    float(data['Height']),
                    float(data['Weight']),
                    float(data['Duration']),
                    float(data['Heart_Rate']),
                    float(data['Body_Temp'])]

    print("Received input values:", int_features)  
    # int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    predicted_list = prediction.tolist()

    return jsonify(predicted_list)
    # return('index.html')

@app.route('/diet',methods=['POST','GET'])
def diet():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        int_features = [float(data['age']),
                    float(data['weight']),
                    float(data['height']),
                    float(data['gender']),
                    float(data['bmi']),
                    float(data['bmr']),
                    float(data['activity_level'])]

        print("Received input values:", int_features)  
        final=[np.array(int_features)]
        dot=Diet.predict(final)
        resp = {
            "prediction":dot[0]
        }

        return jsonify(resp)
    # return render_template('Diet.html')

@app.route('/trainer',methods=['POST','GET'])
def trainer():
    if request.method == 'POST':
        data = request.get_json()
        int_features = [int(data['category']),
                    int(data['sleep']),
                    int(data['recovery']),
                    int(data['body']),]

        print("Received input values:", int_features)  
        final=[np.array(int_features)]
        dot=Trainer.predict(final)
        dotList=dot.astype(int)
        dotList = dotList.tolist()

        predicted_list = {
            "sets":dotList[0][0],
            "reps":dotList[0][1],
            "weights":dotList[0][2],
            "duration":dotList[0][3]
        }

        return jsonify(predicted_list)


