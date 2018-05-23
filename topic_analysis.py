import json
import os
import time

import nltk
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from turnaround_year import turnaround_year

start_year, end_year = 1996, 2016
n_topic_list = [50, 100, 300, 500, 1000]
threshold = 0.2
hot_topic_num_year = 5
hot_topic_num_all = 50


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


def lda_model(data, num_topics, iterations=100, passes=50):
    print(now(), 'training lda model')
    dictionary, corpus = doc2bow(data)
    model = LdaModel(corpus, id2word=dictionary, 
                    num_topics=num_topics, 
                    iterations=iterations, 
                    passes=passes,
                    random_state=42)
    return model


def get_topics(model, num_topics, num_words=10):
    topics = model.show_topics(num_topics=num_topics,
                               num_words=num_words, 
                               formatted=False)
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


def co_occurrence_matrix(has_topic, n_docs, n_topics):
    print(now(), 'calculating co-occurrence matrixes')
    topic_count = has_topic.sum()
    topic_prob = dict(topic_count/n_docs)
    # json.dump(topic_prob, 
            # open('./results/topics/topic occurrence prob.json', 'w'))
    
    # 条件概率，关联
    cond_prob = np.zeros((n_topics, n_topics))
    # 共现
    co_presence = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(n_topics):
            pi, pj = topic_prob[i], topic_prob[j]
            pij = ((has_topic[i]*has_topic[j]).sum())/n_docs
            if pi>0 and pj>0:
                cond_prob[i][j] = pij/pj
                co_presence[i][j] = pij**2/(min(pi, pj)*np.mean([pi, pj]))
            else:
                cond_prob[i][j] = np.nan
                co_presence[i][j] = np.nan
    return co_presence, cond_prob


def hot_topic_all_time(has_topic):
    print(now(), 'finding hot topic of all time')
    hot_topic = has_topic.sum()
    hot_topic = hot_topic.sort_values(ascending=False)
    hot_topic = hot_topic.head(hot_topic_num_all)
    index = list(hot_topic.index)
    occurrence = hot_topic.values
    return index, occurrence


def hot_topics_every_year(has_topic, year):
    print(now(), 'finding hot topic in each year')
    topic_of_the_year = []
    key_papers_num = []
    key_papers_proportion = []
    for y in range(start_year, end_year):
        cond = list(year==y)
        year_topic = has_topic[cond].copy()
        topic_count = year_topic.sum()
        topic_count = topic_count.sort_values(ascending=False)
        year_hot_topic = dict(topic_count[:hot_topic_num_year])
        topic_of_the_year.append(year_hot_topic)
        # print(year_hot_topic)

        year_topic['key'] = year_topic[list(year_hot_topic.keys())].sum(axis=1)
        year_key_papers = (year_topic['key']>0).sum()
        key_papers_num.append(year_key_papers)
        key_papers_proportion.append(year_key_papers/len(year_topic))
        # print(year_key_papers/len(year_topic))

    return topic_of_the_year, key_papers_num, key_papers_proportion


def topic_rk(has_topic, year):
    print(now(), 'calculating topic rk')
    theta_left = []
    theta_right = []
    for y in range(start_year, end_year):
        cond = list(year==y)
        year_topic = has_topic[cond].copy()
        topic_count = year_topic.sum()
        theta = topic_count/(topic_count.sum())
        if y<2006:
            theta_left.append(theta)
        else:
            theta_right.append(theta)
    theta = theta_left + theta_right
    theta_left = np.asarray(theta_left)
    theta_left = theta_left.sum(axis=0)
    theta_right = np.asarray(theta_right)
    theta_right = theta_right.sum(axis=0)
    rk = theta_left/theta_right
    return rk, theta


def topic_analysis(data, type_str, n_topics):
    print(now(), 'analysing {} topics'.format(type_str))
    if not os.path.exists('./results/topics/{}'.format(type_str)):
        os.mkdir('./results/topics/{}'.format(type_str))
    if not os.path.exists('./results/topics/{}/{} topics'.format(type_str, n_topics)):
        os.mkdir('./results/topics/{}/{} topics'.format(type_str, n_topics))
    Dir = './results/topics/{}/{} topics/'.format(type_str, n_topics)

    data = data[data['year'].isin(range(start_year, end_year))].copy()
    text = get_docs(data, title=True, keywords=True, fos=False)
    year = data['year']

    # if type_str=='journal':
    #     n_topics = jrnl_n_topics
    # else:
    #     n_topics = conf_n_topics

    model_file_name = './cache/{} lda model {}'.format(type_str, n_topics)
    if os.path.exists(model_file_name):
        print(now(), 'loading {} lda model n_topics={}'.format(type_str, n_topics))
        model = LdaModel.load(model_file_name)
    else:
        model = lda_model(text, num_topics=n_topics)
        model.save(model_file_name)

    topic_dis = get_topic_distribution(model, text)
    def find_topic(x):
        y = np.zeros(len(x))
        y[np.argmax(x)] = 1
        y[x>threshold] = 1
        return y
    has_topic = topic_dis.apply(find_topic, axis=1)
    hot_topic, occurrence = hot_topic_all_time(has_topic)

    topics, prob = get_topics(model, n_topics)
    topics_df = pd.DataFrame(topics)
    topics_df = topics_df.iloc[hot_topic, :]
    topics_df.to_csv(Dir+'topic terms.csv')
    prob_df = pd.DataFrame(prob)
    prob_df = prob_df.iloc[hot_topic, :]
    prob_df.to_csv(Dir+'topic terms probability.csv')

    hot_topic_df = pd.DataFrame()
    hot_topic_df['topics'] = topics_df.apply(lambda x: ' '.join(x), axis=1)
    hot_topic_df['occurrence'] = occurrence
    rk, theta = topic_rk(has_topic[hot_topic], year)
    hot_topic_df['rk'] = rk
    hot_topic_df.to_csv(Dir+'hot topics.csv')

    co_presence, connection = co_occurrence_matrix(has_topic, len(text), n_topics)
    # connection = pd.DataFrame(connection)
    # connection.to_csv(Dir+'connection.csv')
    co_presence = pd.DataFrame(co_presence)
    co_presence = co_presence.iloc[hot_topic, hot_topic]
    co_presence.to_csv(Dir+'co-presence.csv')

    topic_of_the_year, key_papers_num, key_papers_proportion = hot_topics_every_year(has_topic, year)
    key_papers = pd.DataFrame({
        'topic_of_the_year': [list(item.keys()) for item in topic_of_the_year],
        'occurrence': [list(item.values()) for item in topic_of_the_year],
        'key_papers_num': key_papers_num,
        'key_papers_proportion': key_papers_proportion
    }, index=range(start_year, end_year))
    key_papers.to_csv(Dir+'key papers.csv')

    # turnaround_year(topic_dis, year, type_str, Dir)


if __name__=='__main__':
    if not os.path.exists('./results/topics'):
        os.mkdir('./results/topics')

    for n_topics in n_topic_list:
        jrnl_papers = pd.read_json('./cache/journal_paper_complete.txt')
        conf_papers = pd.read_json('./cache/conference_paper_complete.txt')

        topic_analysis(jrnl_papers, 'journal', n_topics)
        topic_analysis(conf_papers, 'conference', n_topics)
