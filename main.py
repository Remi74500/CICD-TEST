import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time

np.set_printoptions(threshold=10000,suppress=True)
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,ExtraTreesClassifier

import xgboost as xgb


def prepare_and_normalise_dataset():
    df = pd.read_csv('./Churn_Modelling.csv', sep=',', header=0)
    total_rows = len(df.axes[0])
    total_cols = len(df.axes[1])

    #### 1. Séparation du jeu de données (Xtrain) et test (Xtest):

    df.drop(['CustomerId'], axis=1, inplace=True)  # Aucun impact sur la colonne 'Exited'
    X = df.iloc[:, :9].values  # = toutes les colonnes sans la dernière (Exited)
    Y = df.iloc[:, 9].values  # = seulement la dernière colonne

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=1)

    #### 2. Normalisation du jeu de données :

    SS = StandardScaler()
    SS.fit(Xtrain)

    XNormTrain = SS.transform(Xtrain)
    XNormTest = SS.transform(Xtest)

    return XNormTrain, XNormTest, Ytrain, Ytest


def run_classifier(Xtrain, Ytrain, Xtest, Ytest, clfs):
    tot1 = time.time();
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for i in clfs:  # Pour chaque classifieur stocké dans la liste clfs
        clf = clfs[i]
        debut = time.time()

        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)

        acc = accuracy_score(Ytest, Ypred)
        recall = recall_score(Ytest, Ypred)
        mean_score = (acc + recall) / 2

        print("Accuracy for {0} is : {1:.3f} % +/- {2:.3f}".format(i, np.mean(acc) * 100, np.std(acc)))
        print("Recall for {0} is : {1:.3f} % +/- {2:.3f}".format(i, np.mean(recall) * 100, np.std(recall)))
        print("RESULT ====> : {1:.3f} % +/- {2:.3f}".format(i, np.mean(mean_score) * 100, np.std(mean_score)))

        fin = time.time()
        processingTime = fin - debut;
        print("     Execution time for {0} is : {1:.3f} sec".format(i, processingTime))
        print("------------------------------------\n")
    tot2 = time.time();
    totalProcessingTIme = tot2 - tot1
    print(' \n TEMPS TOTAL D EXECUTION : {0:.4f} sec.'.format(totalProcessingTIme));


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest = prepare_and_normalise_dataset()

    clfs = {
        'DT': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'KNN': KNeighborsClassifier(n_neighbors=10),
        'RF': RandomForestClassifier(n_estimators=200, random_state=0),
        'ADA': AdaBoostClassifier(n_estimators=200, random_state=0),
        'ETC': ExtraTreesClassifier(n_estimators=200, criterion='entropy', random_state=0),
        'MLP': MLPClassifier(hidden_layer_sizes=(20, 10), alpha=0.001, max_iter=200),
        'XGB': xgb.XGBClassifier(n_estimators=180, random_state=1, colsample_bytree=0.95, max_depth=2, verbosity=0,
                                 learning_rate=0.1)
    }

    run_classifier(Xtrain, Ytrain, Xtest, Ytest, clfs)





