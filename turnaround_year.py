import os
import time

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


def turnaround_year(topic_dis, year, type_str, Dir):
    print(now(), 'finding {} turnaround year'.format(type_str))

    X = topic_dis
    y = year
    if type_str=='journal':
        test_size=0.2
    else:
        test_size=0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X = scaler.transform(X)
    
    svc = LinearSVC(random_state=42)
    svc.fit(X_train, y_train)

    # param = {'C':np.arange(0.1, 2, 0.2), 'max_iter':np.arange(1000, 10000, 2000)}
    # clf = GridSearchCV(LinearSVC(), param, scoring='accuracy')
    # clf.fit(X_train, y_train)
    # print(now(), 'Best ACC:', clf.best_score_)

    cv_mae = cross_val_score(svc, X_train, y_train, scoring='neg_mean_absolute_error', cv=10)
    cv_mae = np.average(-cv_mae)
    print(now(), 'CV MAE:', cv_mae)

    y_pred = svc.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    print(now(), 'Test MAE:', test_mae)
    print(now(), 'Test ACC:', test_acc)

    y_pred = svc.predict(X)
    # y_pred = clf.predict(X)
    all_mae = mean_absolute_error(y, y_pred)
    all_acc = accuracy_score(y, y_pred)
    print(now(), 'All MAE:', all_mae)
    print(now(), 'All ACC:', all_acc)
    # plot_confusion_matrix(y, y_pred)

    with open(Dir+'score.txt', 'w') as f:
        # f.write('Best param: ')
        # f.write(clf.best_params_)
        # f.write('Best ACC: {}'.format(clf.best_score_))
        f.write('CV MAE: {}\n'.format(cv_mae))
        f.write('Test MAE: {}\n'.format(test_mae))
        f.write('Test ACC: {}\n'.format(test_acc))
        f.write('All MAE: {}\n'.format(all_mae))
        f.write('All ACC: {}\n'.format(all_acc))

    inv_score, year = innovation_score(y, y_pred)
    score = pd.Series(inv_score, index=year)
    score.to_csv(Dir+'innovation score.csv'.format(type_str))
    # plot_innovation_score(inv_score, year)
