from flask import Flask,render_template,request
from pickle import load
import numpy as np
import pandas as pd
from keras.models import load_model

import sklearn
import statsmodels.api as smf
import os


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single')
def single():
    return render_template('single.html')

@app.route('/single_pred', methods=['post'])
def single_ui():
    global loaded_model, prediction, name
    ind = float(request.form.get('ind'))
    man = float(request.form.get('man'))
    ope = float(request.form.get('ope'))
    fin = float(request.form.get('fin'))
    cre = float(request.form.get('cre'))
    com = float(request.form.get('com'))
    classifier = request.form.get('model')
    #print(regressor)
    #loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
    if classifier == 'ada':
        loaded_model = load(open('AdaBoost.sav', 'rb'))
        name = 'AdaBoost Classifier'
    elif classifier == 'random':
        loaded_model = load(open('Random_Forest.sav', 'rb'))
        name = 'Random Forest Classifier'
    elif classifier == 'decision':
        loaded_model = load(open('Decision_Tree.sav', 'rb'))
        name = 'Decision Tree Classifier'
    elif classifier == 'bag':
        loaded_model = load(open('Bagging.sav', 'rb'))
        name = 'Bagging Classifier'
    elif classifier == 'grad':
        loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
        name = 'Gradient Boost Classifier'
    elif classifier == 'stack':
        loaded_model = load(open('Stacking.sav', 'rb'))
        name = 'Stacking Classifier'
    elif classifier == 'knn':
        loaded_model = load(open('KNN.sav', 'rb'))
        name = 'KNN Classifier'
    elif classifier == 'log':
        loaded_model = load(open('Logistic_Regression_model.sav', 'rb'))
        name = 'Logistic Regressor'
    elif classifier == 'nn':
        loaded_model = load_model('ANN.h5')
        name = 'Neural Network Classifier'
    elif classifier == 'rbf':
        loaded_model = load(open('SVC_RBF.sav', 'rb'))
        name = 'SVC - RBF'
    elif classifier == 'poly':
        loaded_model = load(open('SVC_Poly.sav', 'rb'))
        name = 'SVC - Polynomial'
    elif classifier == 'linear':
        loaded_model = load(open('SVC_Linear.sav', 'rb'))
        name = 'SVC - Linear'
    data = {'Industrial Risk': ind,
            'Management Risk': man,
            'Financial Flexibility': fin,
            'Credibility': cre,
            'Competitiveness': com,
            'Operating Risk': ope}
    df = pd.DataFrame(data,index = [0])
    #prediction = (loaded_model.predict(df)[0]).round(6)

    prediction = loaded_model.predict(df)[0]


    data['prediction'] = prediction
    data['name'] = name
    return render_template('single.html', data=data)

@app.route('/multi')
def multi():
    return render_template('multi.html')

@app.route('/multi_pred', methods=['post'])
def multi_ui():
    global loaded_model, prediction, name

    dataset = pd.read_csv(request.form.get('doc'))
    data = dataset[['Industrial Risk', 'Management Risk', 'Financial Flexibility', 'Credibility', 'Competitiveness', 'Operating Risk']]

    classifier = request.form.get('model')
    # print(regressor)
    # loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
    if classifier == 'ada':
        loaded_model = load(open('AdaBoost.sav', 'rb'))
        name = 'AdaBoost Classifier'
    elif classifier == 'random':
        loaded_model = load(open('Random_Forest.sav', 'rb'))
        name = 'Random Forest Classifier'
    elif classifier == 'decision':
        loaded_model = load(open('Decision_Tree.sav', 'rb'))
        name = 'Decision Tree Classifier'
    elif classifier == 'bag':
        loaded_model = load(open('Bagging.sav', 'rb'))
        name = 'Bagging Classifier'
    elif classifier == 'grad':
        loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
        name = 'Gradient Boost Classifier'
    elif classifier == 'stack':
        loaded_model = load(open('Stacking.sav', 'rb'))
        name = 'Stacking Classifier'
    elif classifier == 'knn':
        loaded_model = load(open('KNN.sav', 'rb'))
        name = 'KNN Classifier'
    elif classifier == 'log':
        loaded_model = load(open('Logistic_Regression_model.sav', 'rb'))
        name = 'Logistic Regressor'
    elif classifier == 'nn':
        loaded_model = load_model('ANN.h5')
        name = 'Neural Network Classifier'
    elif classifier == 'rbf':
        loaded_model = load(open('SVC_RBF.sav', 'rb'))
        name = 'SVC - RBF'
    elif classifier == 'poly':
        loaded_model = load(open('SVC_Poly.sav', 'rb'))
        name = 'SVC - Polynomial'
    elif classifier == 'linear':
        loaded_model = load(open('SVC_Linear.sav', 'rb'))
        name = 'SVC - Linear'


    prediction = loaded_model.predict(data)



    data['Class Code'] = prediction
    data['Class'] = np.where(data['Class Code'] == 0, 'Bankrupt', 'Not Bankrupt')
    data.to_csv(f'templates/output/Prediction_{name}.csv')
    result = data.to_csv().encode('utf-8')
    # filename = 'Predictions.csv'
    # save_location = os.path.join('templates', filename)
    # data.save(save_location)
    return render_template('multi.html', data=result)


if __name__ == '__main__':
    app.run(debug=True)