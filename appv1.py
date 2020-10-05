from flask import Flask,request, redirect,jsonify, render_template
import sklearn
import numpy as np
app = Flask(__name__)

from tensorflow.keras.models import load_model
model = load_model('House_price_prediction_model.h5')



@app.route('/')
def home():
    
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features = np.reshape(final_features,(-1, 17))
    prediction = model.predict(final_features)  
	
    if prediction == prediction: a = "The estimated price of the house would be "+ str(prediction) + "$"
    
    return render_template('result.html', Decision ='{}'.format(a))
    

    
    

if __name__ == '__main__':
    app.run()