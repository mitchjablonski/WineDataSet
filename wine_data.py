from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn import svm
from sklearn.model_selection import train_test_split
import model_factory

def predict_wine_rating(df, model_type):
    clf = model_factory.ModelFactory(model_type).factory().build_model()
    output_col = 'quality'
    input_col = list(set(df.columns) - set(output_col))
    X_train, X_test, y_train, y_test = train_test_split(df[input_col], df[output_col], random_state = 45)
    print(set(y_train), set(y_test))
    sm = SMOTE(k_neighbors=3, random_state=12)
    x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    clf.fit(x_train_res, y_train_res)
    print('{} Score for model type {}'.format(clf.score(X_test, y_test), model_type))
    #print('{} Recall Score for model type {}'.format(recall_score(y_test, clf.predict(X_test)), model_type))
def main(wine):
    print(wine)
    df = pd.read_csv('C:\Users\mitch\Desktop\Masters\DataMining2\WineDataSet' + wine,
                   delimiter=';')
    print('Quality Counts {}'.format(np.unique(df['quality'].values, return_counts=True)))
    for model_type in [NEURAL_NET_MODEL, SVM_MODEL, DT_MODEL]:
        predict_wine_rating(df, model_type)

#models
NEURAL_NET_MODEL = 0
SVM_MODEL = 1
#RFC_MODEL = 2
#ETC_MODEL = 3 
#GBC_MODEL = 4  
DT_MODEL = 5
if __name__ =='__main__':
    for wine in [ '\winequality-red.csv', '\winequality-white.csv']:
        main(wine)