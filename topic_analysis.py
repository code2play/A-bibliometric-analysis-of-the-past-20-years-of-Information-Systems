import json
import os
import time

import lda
import nltk
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words

from turnaround_year import turnaround_year

start_year, end_year = 1996, 2016
n_topics = 100
threshold = 0.2


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def get_docs(data, title=True, keywords=True, fos=False):
    def pad_space(x):
        if pd.isna(x):
            return ' '
        return ' ' + x + ' '
    def concat(x):
        if x==None:
            return ' '
        return ' ' + ' '.join(x) + ' '
    data = data.copy()
    data['title'] = data['title'].apply(pad_space)
    data['abstract'] = data['abstract'].apply(pad_space)
    data['keywords'] = data['keywords'].apply(concat)
    data['fos'] = data['fos'].apply(concat)

    data['text'] = data['abstract']
    if title:
        data['text'] += data['title']
    if keywords:
        data['text'] += data['keywords']
    if fos:
        data['text'] += data['fos']
    data['text'] = data['text'].apply(str.lower)
    return np.array(data['text'])


def doc2bow(data):
    # Tokenize
    tokenizer = RegexpTokenizer('\w[\w\'\-]+')
    tokens = [tokenizer.tokenize(text) for text in data]

    # Exclud stop words
    # en_stop = get_stop_words('en')
    # nltk.download('stopwords')
    en_stop = stopwords.words('english')
    tokens = [[j for j in i if j not in en_stop] for i in tokens]

    # Stemming
    # nltk.download('wordnet')
    wnl = WordNetLemmatizer()
    tokens = [[wnl.lemmatize(j, pos='n') for j in i]for i in tokens]
    tokens = [[wnl.lemmatize(j, pos='v') for j in i]for i in tokens]

    # Doc2bow
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(word) for word in tokens]

    return dictionary, corpus


def lda_model(data, num_topics=20, iterations=100, passes=50):
    print(now(), 'training lda model')
    dictionary, corpus = doc2bow(data)
    model = LdaModel(corpus, id2word=dictionary, 
                    num_topics=num_topics, 
                    iterations=iterations, 
                    passes=passes,
                    random_state=42)
    return model


def get_topics(model, num_topics=20):
    topics = model.show_topics(num_topics=num_topics, formatted=False)
    topic_list = [[j[0] for j in i[1]] for i in topics]
    topic_prob = [[j[1] for j in i[1]] for i in topics]
    # topics = [' '.join(t) for t in topic_list]
    return topic_list, topic_prob


def get_topic_distribution(model, data):
    dictionary, corpus = doc2bow(data)
    topics = model.get_document_topics(corpus, minimum_probability=0)
    topic_distribution = [{j[0]:j[1] for j in i} for i in topics]
    return pd.DataFrame(topic_distribution)


def get_term_distribution(model, data):
    pass


def co_occurrence_matrix(has_topic, n_docs):
    print(now(), 'calculating co-occurrence matrixes')
    topic_count = has_topic.sum()
    topic_prob = dict(topic_count/n_docs)
    json.dump(topic_prob, 
            open('./results/topics/topic occurrence prob.json', 'w'))
    
    # 条件概率，关联
    cond_prob = np.zeros((n_topics, n_topics))
    # 共现
    co_presence = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(n_topics):
            # ii, jj = str(i), str(j)
            pi, pj = topic_prob[i], topic_prob[j]
            pij = ((has_topic[i]*has_topic[j]).sum())/n_docs
            if pi>0 and pj>0:
                cond_prob[i][j] = pij/pj
                co_presence[i][j] = pij**2/(min(pi, pj)*np.mean([pi, pj]))
            else:
                cond_prob[i][j] = np.inf
                co_presence[i][j] = np.inf
    return cond_prob, co_presence


def popular_topics(has_topic, year):
    print(now(), 'finding popular topics')
    theta_left = []
    theta_right = []
    for y in range(start_year, end_year):
        cond = list(year==y)
        year_topic = has_topic[cond]
        topic_count = year_topic.sum()
        theta = topic_count/(topic_count.sum())
        if y<2006:
            theta_left.append(theta)
        else:
            theta_right.append(theta)
    theta_left = np.asarray(theta_left)
    theta_left = theta_left.sum(axis=0)
    theta_right = np.asarray(theta_right)
    theta_right = theta_right.sum(axis=0)
    rk = theta_left/theta_right
    return rk


def topic_analysis(data, type_str):
    print(now(), 'analysing {} topics'.format(type_str))
    if not os.path.exists('./results/topics/{}'.format(type_str)):
        os.mkdir('./results/topics/{}'.format(type_str))
    Dir = './results/topics/{}/'.format(type_str)

    data = data[data['year'].isin(range(start_year, end_year))].copy()
    text = get_docs(data, title=True, keywords=True, fos=False)
    year = data['year']

    model_file_name = './cache/{} lda model'.format(type_str)
    if os.path.exists(model_file_name):
        model = LdaModel.load(model_file_name)
    if model.num_topics!=n_topics:
        model = lda_model(text, num_topics=n_topics)
        model.save(model_file_name)

    topics, prob = get_topics(model, n_topics)
    topics = pd.DataFrame(topics)
    prob = pd.DataFrame(prob)
    topics.to_csv(Dir+'topic terms.csv')
    prob.to_csv(Dir+'topic terms probability.csv')

    topic_dis = get_topic_distribution(model, text)
    def find_topic(x):
        y = np.zeros(len(x))
        y[np.argmax(x)] = 1
        y[x>threshold] = 1
        return y
    has_topic = topic_dis.apply(find_topic, axis=1)

    connection, co_presence = co_occurrence_matrix(has_topic, len(text))
    connection = pd.DataFrame(connection)
    connection.to_csv(Dir+'connection.csv')
    co_presence = pd.DataFrame(co_presence)
    co_presence.to_csv(Dir+'co-presence.csv')

    rk = popular_topics(has_topic, year)
    rk = pd.Series(rk)
    rk.to_csv(Dir+'Rk.csv')

    turnaround_year(topic_dis, year, type_str)


if __name__=='__main__':
    if not os.path.exists('./results/topics'):
        os.mkdir('./results/topics')

    jrnl_papers = pd.read_json('./cache/journal_paper_complete.txt')
    conf_papers = pd.read_json('./cache/conference_paper_complete.txt')

    topic_analysis(jrnl_papers, 'journal')
    topic_analysis(conf_papers, 'conference')
