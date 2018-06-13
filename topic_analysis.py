import json
import os
import time

import langid
import nltk
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from plotter import plot_corr_matrix, plot_word_cloud
from turnaround_year import turnaround_year

start_year, end_year = 1996, 2016
n_topic_list = [50, 100, 300, 500, 1000]
threshold = 0.2
hot_topic_num_year = 5
hot_topic_num_all = 50


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def get_docs(data, title=True, keywords=False):
    kws = [j for i,v in data['keywords'].iteritems() if v!=None for j in v]
    kws = pd.unique(kws)

    def pad_space(x):
        if pd.isna(x):
            return ' '
        return ' ' + x + ' '
    def concat(x):
        if x==None:
            return ' '
        # x = [i for i in x if langid.classify(i)[0]=='en']
        return ' ' + ' '.join(x) + ' '
    data['title'] = data['title'].apply(pad_space)
    data['abstract'] = data['abstract'].apply(pad_space)
    data['keywords'] = data['keywords'].apply(concat)

    data['text'] = data['abstract']
    if title:
        data['text'] += data['title']
    if keywords:
        data['text'] += data['keywords']
    data['text'] = data['text'].apply(str.lower)

    def is_en(x):
        if pd.isna(x):
            return False
        return langid.classify(x)[0]=='en'
    cond = data['year'].isin(range(start_year, end_year)) & \
           data['text'].apply(is_en)
    
    return data[cond][['text', 'year']], kws

# 效果不好，匹配的单词出现频率太低
def match_keywords(tokens, kw_list):
    wnl = WordNetLemmatizer()
    prefix = []
    keywords = []
    for k in kw_list:
        kws = k.split(' ')
        temp = []
        for i in range(0,len(kws)):
            word = wnl.lemmatize(kws[i], pos='n')
            temp.append(word)
            if i>0:
                prefix.append(' '.join(temp[:i]))
        keywords.append(' '.join(temp))
    prefix = dict.fromkeys(prefix)
    keywords = dict.fromkeys(keywords)

    matched_tokes = []
    cnt = 0
    for t in tokens:
        temp = []
        i = 0
        while(i<len(t)):
            def dfs(b, e):
                s = ' '.join(t[b:e])
                if s not in prefix or e==len(t):
                    if s in keywords:
                        return e
                    else:
                        return -1
                x = dfs(b, e+1)
                if x>0:
                    return x
                elif s in keywords:
                    return e
                return -1
            j = dfs(i, i+1)
            if j>0:
                temp.append(' '.join(t[i:j]))
                cnt += 1
                i = j
            else:
                temp.append(t[i])
                i += 1
        matched_tokes.append(temp)
    print(now(), 'matched {} keywords'.format(cnt))
    return matched_tokes


def doc2bow(text, keywords, match=False):
    # Tokenize
    tokenizer = RegexpTokenizer('[a-z][a-z]+')
    tokens = [tokenizer.tokenize(t) for t in text]

    # Stemming
    # nltk.download('wordnet')
    wnl = WordNetLemmatizer()
    tokens = [[wnl.lemmatize(j, pos='n') for j in i]for i in tokens]
    if match:
        match_keywords(tokens, keywords)
    tokens = [[wnl.lemmatize(j, pos='v') for j in i]for i in tokens]

    # Exclud stop words
    # nltk.download('stopwords')
    # en_stop = stopwords.words('english')
    en_stop = json.load(open('./cache/en.json', 'r'))
    en_stop = dict.fromkeys(en_stop)
    tokens = [[j for j in i if j not in en_stop] for i in tokens]

    es_stop = json.load(open('./cache/es.json', 'r', encoding='UTF-8'))
    es_stop = dict.fromkeys(es_stop)
    tokens = [[j for j in i if j not in es_stop] for i in tokens]

    # Doc2bow
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(word) for word in tokens]

    return dictionary, corpus


def lda_model(corpus, dictionary, num_topics, model_file_name):
    if os.path.exists(model_file_name):
        print(now(), 'loading lda model n_topics={}'.format(num_topics))
        model = LdaModel.load(model_file_name)
    else:
        print(now(), 'training lda model n_topics={}'.format(num_topics))
        model = LdaMulticore(corpus, 
                            id2word=dictionary, 
                            num_topics=num_topics, 
                            iterations=100, 
                            passes=50,
                            random_state=42)
        model.save(model_file_name)
    return model


def get_topics(model, num_topics, num_words=10):
    topics = model.show_topics(num_topics=num_topics,
                               num_words=num_words, 
                               formatted=False)
    topic_list = [[j[0] for j in i[1]] for i in topics]
    topic_prob = [[j[1] for j in i[1]] for i in topics]
    # topics = [' '.join(t) for t in topic_list]
    return topic_list, topic_prob


def get_topic_distribution(model, corpus):
    topics = model.get_document_topics(corpus, minimum_probability=0)
    topic_distribution = [{j[0]:j[1] for j in i} for i in topics]
    return pd.DataFrame(topic_distribution)


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
    for i, ti in enumerate(has_topic.columns):
        for j, tj in enumerate(has_topic.columns):
            pi, pj = topic_prob[ti], topic_prob[tj]
            pij = ((has_topic[ti]*has_topic[tj]).sum())/n_docs
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
    theta = np.asarray(theta_left + theta_right)
    theta_left = np.asarray(theta_left)
    theta_left = theta_left.sum(axis=0)
    theta_right = np.asarray(theta_right)
    theta_right = theta_right.sum(axis=0)
    rk = theta_right/theta_left
    return rk, theta


def topic_analysis(data, keywords, type_str, n_topics):
    print(now(), 'analysing {} topics'.format(type_str))
    if not os.path.exists('./results/topics/{}'.format(type_str)):
        os.mkdir('./results/topics/{}'.format(type_str))
    if not os.path.exists('./results/topics/{}/{} topics'.format(type_str, n_topics)):
        os.mkdir('./results/topics/{}/{} topics'.format(type_str, n_topics))
    Dir = './results/topics/{}/{} topics/'.format(type_str, n_topics)

    text = list(data['text'])
    year = data['year']

    dictionary, corpus = doc2bow(text, keywords, match=False)
    model_file_name = './cache/{} lda model {}'.format(type_str, n_topics)
    model = lda_model(corpus, dictionary, n_topics, model_file_name)

    topic_dis = get_topic_distribution(model, corpus)
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
    plot_word_cloud(topics_df, prob_df, Dir)

    hot_topic_df = pd.DataFrame()
    hot_topic_df['topics'] = topics_df.apply(lambda x: ' '.join(x), axis=1)
    hot_topic_df['occurrence'] = occurrence
    rk, theta = topic_rk(has_topic[hot_topic], year)
    hot_topic_df['rk'] = rk
    hot_topic_df.to_csv(Dir+'hot topics.csv')
    theta_df = pd.DataFrame(theta, 
                            index=range(start_year, end_year), 
                            columns=hot_topic)
    theta_df.to_csv(Dir+'theta.csv')

    co_presence, connection = co_occurrence_matrix(has_topic[hot_topic], 
                                                   len(text), 
                                                   hot_topic_num_all)
    # connection = pd.DataFrame(connection)
    # connection.to_csv(Dir+'connection.csv')
    co_presence = pd.DataFrame(co_presence,
                               index=hot_topic,
                               columns=hot_topic)
    co_presence.to_csv(Dir+'co-presence.csv')
    plot_corr_matrix(co_presence, Dir)

    topic_of_the_year, key_papers_num, key_papers_proportion = hot_topics_every_year(has_topic, year)
    key_papers = pd.DataFrame({
        'topic_of_the_year': [list(item.keys()) for item in topic_of_the_year],
        'occurrence': [list(item.values()) for item in topic_of_the_year],
        'key_papers_num': key_papers_num,
        'key_papers_proportion': key_papers_proportion
    }, index=range(start_year, end_year))
    key_papers.to_csv(Dir+'key papers.csv')

    turnaround_year(topic_dis, year, type_str, Dir)


if __name__=='__main__':
    if not os.path.exists('./results/topics'):
        os.mkdir('./results/topics')

    jrnl_papers = pd.read_json('./cache/journal_papers.txt')
    jrnl_papers, jrnl_kws = get_docs(jrnl_papers, title=True, keywords=False)

    conf_papers = pd.read_json('./cache/conference_papers.txt')
    conf_papers, conf_kws = get_docs(conf_papers, title=True, keywords=False)

    for n_topics in n_topic_list:
        topic_analysis(jrnl_papers, jrnl_kws, 'journal', n_topics)
        topic_analysis(conf_papers, conf_kws, 'conference', n_topics)
