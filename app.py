from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import pickle

# ML packages from Sci Kit learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

app= Flask(__name__)
Bootstrap(app)
# log_model= pickle.load(open('joblib_sample.sav'),'rb')

@app.route('/')
def index():
    return render_template('index_1.html')

@app.route('/predict',methods= ['POST'])
def predict():
    df= pd.read_csv("logistic_regression_updated.csv")
    array= df.values
    X= array[:,0:6]
    Y= array[:,6]
    filename= 'models/finalized_model_joblib_1.sav'
    regressor = LogisticRegression()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    classifier = regressor.fit(X,Y)
    _=joblib.dump(regressor,filename)
    joblib_model = joblib.load(filename)
    
    #model = open(filename,"rb")
  
    # prediction_model = joblib.load(model)
    
    # model = open('pickle_sample.pkl','rb')
    
    
   

    
    
    
   # df= pd.read_csv('standardised_values_300.csv')
    #df_x= [df.drop(['Rising_Star'],axis=1)]
    #df_y = df['Rising_Star']
     
        
    if request.method == 'POST':
    
        age = request.form['age']
        games_played = request.form['games_played']
        games_started = request.form['games_started']
        assist_pct = request.form['assist_pct']
        usage_pct = request.form['usage_pct']
        def_rtg = request.form['def_rtg']
        data = [[age,games_played,games_started,assist_pct,usage_pct,def_rtg]]
        data_transform = scaler.transform(data)
        data_transform_list = np.array(data_transform)
        
        #float_data_list = list.astype(float)
        #scaler = StandardScaler()
        #transform_player_list= scaler.fit(float_data_list)
        #transform_player_list=  scaler.transform(transform_player_list)
        
        #to_array = np.asarray(data_list)
        #shape= data_list.reshape(1,-1)
        
        #shape = np.shape()
        #my_prediction = joblib_model.predict(transform_player_list)
        my_prediction = joblib_model.predict(data_transform_list)
        
     
        
    return render_template('results_1.html',prediction = my_prediction)
    
    
   
    


    
    
    

if __name__ == '__main__':
    app.run(debug = True)
    
    