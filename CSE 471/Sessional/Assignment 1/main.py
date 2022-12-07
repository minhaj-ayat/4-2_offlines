import math
import random

import numpy
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Number_of_Features = 10
Max_iter = 50
Alpha = 0.01
X_train = pd.DataFrame()
y_train = pd.DataFrame()
X_test = pd.DataFrame()
y_test = pd.DataFrame()
selected_features_dataX = pd.DataFrame()
selected_features_testX = pd.DataFrame()


def preprocess(fn):
    data = pd.read_csv(fn)
    data.drop(['customerID'], 1, inplace=True)
    #print(data)
    #print(data.dtypes)
    string_columns = []
    yes_no = {'Yes', 'yes', 'No', 'no', 'Male', 'Female'}
    for col in data:
        unique_values = data[col].unique()
        #print(len(unique_values))
        if len(unique_values) == 2 and any([i in unique_values for i in yes_no]):
            #print(col, 'has yes/no value')
            if col != "Churn":
                data[col].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}, inplace=True)
            else:
                data[col].replace({'Yes': 1, 'No': -1, 'Male': 1, 'Female': -1}, inplace=True)
        elif data[col].dtype == "int64" or data[col].dtype == "float64":
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(0)
            max = (data[col].max())
            min = (data[col].min())
            #print("Max:", max, ':::::  Min:', min)
            data[col] = ((data[col] - min) / (max - min))
        elif data[col].dtype == "object" and len(unique_values) > 2:
            if len(unique_values) <= 4:
                #print(len(unique_values))
                string_columns.append(col)
            else:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(0)
                max = (data[col].max())
                min = (data[col].min())
                #print("Max:", max , ':::::  Min:',min)
                data[col] = ((data[col]-min)/(max-min))

    for c in string_columns:
        data = pd.get_dummies(data, c, drop_first=True)
    print("Actual Data--")
    print(data)
    data.to_csv("out.csv", sep='\t')

    X = data.drop(['Churn'],axis=1)
    y = data['Churn']
    global X_train,X_test,y_train,y_test,selected_features_dataX,selected_features_testX
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print("\ny_train--")
    print(X_train)

    selector = SelectKBest(score_func=mutual_info_classif, k=Number_of_Features)
    selector.fit(X_train, y_train)
    cols = selector.get_support(indices=True)
    selected_features_dataX = X_train.iloc[:, cols]
    selected_features_testX = X_test.iloc[:, cols]
    print("\nBest feature--")
    print(cols)
    print(selected_features_dataX)
    return data


def logistic_regression(sfd_matrix, ytrain):
    print("\nIn logistic regression")
    w = []
    for i in range(0,Number_of_Features):
        w.append(random.uniform(-1,1))
    row_count = sfd_matrix.shape[0]
    #print("\nTrain data rows - ", row_count)

    y_train_matrix = ytrain.to_numpy()
    #print("\ntrain data last row - \n", sfd_matrix[row_count-1])
    #print("\nTrain output no.8 : \n", y_train_matrix[8])
    #print("\nTrain data cell [1][0]:\n", sfd_matrix[1][0])
    #print("\nShapes", w, sfd_matrix[6],y_train_matrix)
    #print(sfd_matrix[row_count-1])

    for k in range(1, Max_iter):
        for i in range(0, row_count):
            wx = np.dot(w, sfd_matrix[i])
            hx = np.tanh(wx)
            sum = (y_train_matrix[i] - hx) * (1 - (hx * hx))
            for j in range(0, Number_of_Features):
                w[j] += Alpha * sum * sfd_matrix[i][j]
    print("\nWeights:\n", w)
    return w


def printResults(w):
    selected_features_testX_matrix = selected_features_testX.to_numpy()
    y_test_matrix = y_test.to_numpy()
    row_count_test = selected_features_testX.shape[0]

    success = 0
    predict1 = 0
    predict0 = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, row_count_test):
        wx = np.dot(w, selected_features_testX_matrix[i])
        hx = np.tanh(wx)
        # hx = 1/(1 + np.exp(-wx))
        if hx >= 0:
            ans = 1.0
            predict1 += 1
        else:
            ans = -1.0
            predict0 += 1
        if ans == y_test_matrix[i]:
            success += 1
        if ans == 1.0 and y_test_matrix[i] == 1.0:
            tp += 1
        if ans == 1.0 and y_test_matrix[i] == -1.0:
            fp += 1
        if ans == -1.0 and y_test_matrix[i] == -1.0:
            tn += 1
        if ans == -1.0 and y_test_matrix[i] == 1.0:
            fn += 1

    print("\n---Test data----")
    print("Success : ", success)
    print("Predict 1 : ", predict1)
    print("Predict -1 : ", predict0)
    print("Test data rows: ", row_count_test)
    print("Accuracy rate : ", success / row_count_test,"\n")
    print("True positive rate : ", tp/(tp+fn), "\n")
    print("True negative rate : ", tn / (tn + fp), "\n")
    print("Precision  : ", tp / (tp + fp), "\n")
    print("False discovery rate : ", fp / (fp + tp), "\n")
    print("F1 score : ", 2*tp / (2*tp + fp + fn), "\n")

    selected_features_dataX_matrix = selected_features_dataX.to_numpy()
    y_train_matrix = y_train.to_numpy()
    row_count_train = selected_features_dataX.shape[0]
    success = 0
    predict1 = 0
    predict0 = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, row_count_test):
        wx = np.dot(w, selected_features_dataX_matrix[i])
        hx = np.tanh(wx)
        # hx = 1/(1 + np.exp(-wx))
        if hx >= 0:
            ans = 1.0
            predict1 += 1
        else:
            ans = -1.0
            predict0 += 1
        if ans == y_train_matrix[i]:
            success += 1
        if ans == 1.0 and y_train_matrix[i] == 1.0:
            tp += 1
        if ans == 1.0 and y_train_matrix[i] == -1.0:
            fp += 1
        if ans == -1.0 and y_train_matrix[i] == -1.0:
            tn += 1
        if ans == -1.0 and y_train_matrix[i] == 1.0:
            fn += 1

    print("\n---Training data----")
    print("Success : ", success)
    print("Predict 1 : ", predict1)
    print("Predict -1 : ", predict0)
    print("Test data rows: ", row_count_train)
    print("Accuracy rate : ", success / row_count_train, "\n")
    print("True positive rate : ", tp / (tp + fn), "\n")
    print("True negative rate : ", tn / (tn + fp), "\n")
    print("Precision  : ", tp / (tp + fp), "\n")
    print("False discovery rate : ", fp / (fp + tp), "\n")
    print("F1 score : ", 2 * tp / (2 * tp + fp + fn), "\n")


def adaboost(K):
    N = len(X_train)
    example_weights = [1/N] * N
    hypotheses = []
    final = []
    #print(hypotheses)
    z = []

    sfd = selected_features_dataX.to_numpy()
    y_train_matrix = y_train.to_numpy()
    for k in range(0, K):
        print("In adaboost round :", k)
        selected_rows = np.random.choice(sfd.shape[0], N, p=example_weights, replace=True)
        dataX = sfd[selected_rows]
        dataY = y_train_matrix[selected_rows]
        print(dataX.shape)
        hypotheses.append(logistic_regression(dataX, y_train))
        #print(hypotheses)
        error = 0
        for j in range(0, N):
            wx = np.dot(hypotheses[k], dataX[j])
            hx = np.tanh(wx)
            ans = 1 if hx >= 0 else -1
            if ans != dataY[j]:
                error += example_weights[j]
        if error > 0.5:
            break
        for j in range(0, N):
            wx = np.dot(hypotheses[k], dataX[j])
            hx = np.tanh(wx)
            ans = 1 if hx >= 0 else -1
            if ans == dataY[j]:
                example_weights[j] = example_weights[j] * (error/(1-error))

        minm = min(example_weights)
        maxm = max(example_weights)
        s = sum(example_weights)
        for ew in range(0, N):
            example_weights[ew] = example_weights[ew]/s

        z.append(math.log2((1-error)/error))
    print("Z:\n", z)
    for i in range(0, K):
        hypotheses[i] = [element * z[i] for element in hypotheses[i]]
    print(hypotheses)
    for j in range(0, Number_of_Features):
        fsum = 0
        for i in range(0, K):
            fsum += hypotheses[i][j]
        final.append(fsum)
    print(final)
    printResults(final)


filename = "telco.csv"
df = preprocess(filename)
selected_features_dataX_matrix = selected_features_dataX.to_numpy()
weigths = logistic_regression(selected_features_dataX_matrix, y_train)
printResults(weigths)
#adaboost(5)
