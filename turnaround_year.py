import os
import time

import lda
import nltk
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_absolute_error)
from sklearn.model_selection import (GridSearchCV, KFold,
                                     StratifiedShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from stop_words import get_stop_words


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)
    n = len(labels)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
            rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            if cm[i][j]==0:
                continue
            text = ax.text(j, i, cm[i, j],
                        ha="center", va="center", color="w")
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def innovation_score(y_true, y_pred):
    year = np.unique(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    condf = y_true<y_pred
    condp = y_true>y_pred
    future = y_pred[condf]
    past = y_pred[condp]
    score = []
    for y in year:
        condfy = condf & (y_true==y)
        condpy = condp & (y_true==y)
        errf = (y_pred[condfy] - y_true[condfy]).sum()
        errp = (y_true[condpy] - y_pred[condpy]).sum()
        py = (y_true==y).sum()

        NF = (y-future.min())/(future.max()-future.min())
        NP = (past.max()-y)/(past.max()-past.min())
        # NF = NP = 1

        score.append(errf/py*NF - errp/py*NP)
    return score, year


def plot_innovation_score(score, year):
    fig, ax = plt.subplots()
    ax.set_xticks(year)
    ax.set_xticklabels(year)
    plt.bar(year, score)
    plt.xlabel('Year')
    plt.ylabel('Innovation Score')
    plt.show()


def turnaround_year(topic_dis, year, type_str):
    print(now(), 'finding {} turnaround year'.format(type_str))

    X = topic_dis
    y = year
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X = scaler.transform(X)
    
    svc = LinearSVC(random_state=42)
    svc.fit(X_train, y_train)

    # score = cross_val_score(svc, X, y, scoring='neg_mean_absolute_error', 
                            # cv=KFold(10))
    # score = np.average(-score)
    # print('CV MAE:', score)

    y_pred = svc.predict(X)
    print('Test MAE:', mean_absolute_error(y, y_pred))
    print('Test ACC:', accuracy_score(y, y_pred))

    # param = {'C':np.arange(0.1, 3, 0.2), 'max_iter':np.arange(1000, 10000, 1000)}
    # clf = GridSearchCV(LinearSVC(), param, scoring='accuracy')
    # clf.fit(X, y)
    # print(clf.best_score_)
    # y_pred = clf.predict(X)

    # plot_confusion_matrix(y, y_pred)

    inv_score, year = innovation_score(y, y_pred)
    score = pd.Series(inv_score, index=year)
    score.to_csv('./results/topics/{}/innovation score.csv'.format(type_str))
    # plot_innovation_score(inv_score, year)
