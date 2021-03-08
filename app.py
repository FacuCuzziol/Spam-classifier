from flask import Flask,render_template,url_for,request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})


@app.route('/',methods=['POST','GET'])
def home():
	return render_template('index.html')

@app.route('/predictspam',methods=['POST','GET'])
def predictspam():

    error =''
    
    predict_result =''

    data = request.get_json()
    message = data['inputt_1']
#    message = request.form['message']
    vectorized_message = vectorizer.transform([message])
    print('MODEL',cc_model)
    predict_result = cc_model.predict(vectorized_message)[0]
    #predict_proba = trained_model.predict_proba(vectorized_message).tolist()
    if( predict_result ==0):
        print('ham')
        predict_result = 'Ham'
    elif (predict_result ==1):
        print('spam')
        predict_result ='Spam'
    #return render_template('index.html',prediction=predict)
    data_res = {
        'result_cat':predict_result
    }
    return data_res




if __name__ == "__main__":
    print("******* Running Model ********")


    #load the model
    trained_model = 'data/calibrated_classifier_03032021.pkl'
    vectorizer_file = 'data/tfid_vectorizer_03032021.pkl'
    with open(trained_model,'rb') as f:
        cc_model = pickle.load(f)
        print('model_loaded')
    with open(vectorizer_file,'rb') as fl:
        vectorizer = pickle.load(fl)


    app.run(debug=False,use_reloader=False)