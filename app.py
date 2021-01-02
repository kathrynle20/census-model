from flask import *
import pickle

import pandas as pd
import numpy as np

# Use pickle to load in the pre-trained model.
with open(f'model/census_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    #return ("Hello World")
    #return(render_template('Census-model/main.html'))
    return render_template('main.html')

def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 12) 
    result = model.predict(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='an income of more than 50K'
        else: 
            prediction ='an income of less than 50K'            
        return render_template("result.html", prediction = prediction) 

if __name__ == '__main__':
    app.run()