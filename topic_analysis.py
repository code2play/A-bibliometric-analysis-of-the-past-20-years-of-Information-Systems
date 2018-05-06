import os
import time

import lda
import nltk
import pandas as pd
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from scipy.spatial.distance import cosine, euclidean
from stop_words import get_stop_words


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def read_data():
    venues = []
    with open('./journals.txt', 'r') as f:
        for line in f:
            name, real_name = line.split('\t')
            venues.append(real_name.strip().lower())
    with open('./conferences.txt', 'r') as f:
        for line in f:
            name, real_name = line.split('\t')
            venues.append(real_name.strip().lower())

    jrnl_papers = pd.read_json('./cache/journal_paper_complete.txt')
    conf_papers = pd.read_json('./cache/conference_paper_complete.txt')
    papers = pd.concat([jrnl_papers, conf_papers])
    return papers, venues


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
    tokens = [[wnl.lemmatize(j) for j in i]for i in tokens]

    # Doc2bow
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(word) for word in tokens]

    return dictionary, corpus


def lda_model(data, num_topics=20, iterations=100, passes=50):
    dictionary, corpus = doc2bow(data)
    model = LdaModel(corpus, id2word=dictionary, 
                    num_topics=num_topics, 
                    iterations=iterations, 
                    passes=passes,
                    random_state=42)
    return model


def get_topics(model, data):
    dictionary, corpus = doc2bow(data)
    topics = model.show_topics(formatted=False)
    topic_list = [[j[0] for j in i[1]] for i in topics]
    topics = [' '.join(t) for t in topic_list]
    return topics


def get_topic_distribution(model, data):
    dictionary, corpus = doc2bow(data)
    topics = model.get_document_topics(corpus, 
                    minimum_probability=0)
    topic_distribution = [{j[0]:j[1] for j in i} for i in topics]
    return topic_distribution


def get_term_distribution():
    pass


def top_cited_every_year(data, top=30, start_year=1996, end_year=2016):
    for y in range(start_year, end_year):
        cond = data['year']==y
        if y==start_year:
            train = data[cond].head(top)
        else:
            train = pd.concat([train, data[cond].head(top)])
    return train


if __name__=='__main__':
    if not os.path.exists('./results/topics'):
        os.mkdir('./results/topics')

    start_year, end_year = 1996, 2016
    papers, venues = read_data()

    for venue in venues:
        result = pd.DataFrame({'venue':[],'year':[],'topic':[]})
        for year in range(start_year, end_year):
            print(now(), venue, year)

            cond = papers['abstract'].notna() & \
                    (papers['real_venue']==venue) & \
                    (papers['year']==year)
            abstract = list(papers[cond]['abstract'].apply(str.lower))
            if len(abstract)==0:
                print(now(), 'No papers found')
                continue

            num_topics = 5
            topics = get_topics(abstract, num_topics)

            df = pd.DataFrame({
                'venue': [venue for i in range(num_topics)],
                'year': [year for i in range(num_topics)],
                'topic': topics
            })
            result = pd.concat([result, df])
        result.to_csv('./results/topics/{} topics.csv'.format(venue))
