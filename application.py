#importing libraries
from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

#creating Flask app
app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('HeartFailureModel.pkl','rb'))
dataset=pd.read_csv('heart_disease_health_indicators (1).csv')

#setting route for get and post methods
@app.route('/',methods=['GET','POST'])
def index():
    

    highbp=dataset['HighBP'].unique()
    highchol=dataset['HighChol'].unique()
    cholcheck = dataset['CholCheck'].unique()
    bmi = dataset['BMI'].unique()
    smoker = dataset['Smoker'].unique()
    stroke = dataset['Stroke'].unique()
    diabetes = dataset['Diabetes'].unique()
    physactivity = dataset['PhysActivity'].unique()
    fruits = dataset['Fruits'].unique()
    veggies = dataset['Veggies'].unique()
    hvyalcoholconsump = dataset['HvyAlcoholConsump'].unique()
    anyhealthcare = dataset['AnyHealthcare'].unique()
    nodocbccost = dataset['NoDocbcCost'].unique()
    genhlth = dataset['GenHlth'].unique()
    menthlth = dataset['MentHlth'].unique()
    physhlth = dataset['PhysHlth'].unique()
    diffwalk = dataset['DiffWalk'].unique()
    sex = dataset['Sex'].unique()
    age = dataset['Age'].unique()
    education = dataset['Education'].unique()
    income = dataset['Income'].unique()
    return render_template('index.html', )


#setting route for predict method
@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    highBp= int(request.form.get('HighBp'))
    highChol = int(request.form.get("HighChol"))
    cholCheck = int(request.form.get("CholCheck"))
    bmi = int(request.form.get("BMI"))
    smoker = int(request.form.get("Smoker"))
    stroke = int(request.form.get("Stroke"))
    diabetes = int(request.form.get("Diabetes"))
    physActivity =int( request.form.get("PhysActivity"))
    fruits = int(request.form.get("Fruits"))
    veggies = int(request.form.get("Veggies"))
    hvyAlcoholConsump = int(request.form.get("HvyAlcoholConsump"))
    anyHealthcare = int(request.form.get("AnyHealthcare"))
    noDocbcCost =int( request.form.get("NoDocbcCost"))
    genHlth =int( request.form.get("GenHlth"))
    mentHlth =int( request.form.get("MentHlth"))
    physHlth = int(request.form.get("PhysHlth"))
    diffWalk = int(request.form.get("DiffWalk"))
    sex = int(request.form.get("sex"))
    age = int(request.form.get("Age"))
    education =int(request.form.get("Education"))
    income = int(request.form.get("Income"))
    
   
    
    prediction=model.predict(pd.DataFrame(columns=['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','Diabetes','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk',"Sex","Age","Education","Income"],
                              data=np.array([highBp,highChol,cholCheck,bmi,smoker,stroke,diabetes,physActivity,fruits,veggies,hvyAlcoholConsump,anyHealthcare,noDocbcCost,genHlth,mentHlth,physHlth,diffWalk,sex,age,education,income ] ).reshape(1, 21)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()