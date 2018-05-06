import json
import os
import time

import numpy as np
import pandas as pd

start_year, end_year = 1996, 2016


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def top_papers(data, type_str):
    print(now(), 'finding top papers')
    data = data[data['year'].isin(range(start_year, end_year))]
    data = data.sort_values('citations', ascending=False)
    top = data.head(30)[['title', 'citations', 'year']]
    top.index = range(1, 31)
    top.to_csv('./results/top30/top {} papers.csv'.format(type_str))


def top_authors_and_orgs(data):
    print(now(), 'finding top authors and orgs')
    data = data[data['year'].isin(range(start_year, end_year))]
    authors = list(data['authors'])
    authors = [json.dumps(j) for i in authors for j in i]
    uniq_authors = pd.unique(authors)
    uniq_authors = [json.loads(i) for i in uniq_authors]

    names = []
    orgs = []
    citations = []

    for author in uniq_authors:
        names.append(author['name'])
        if 'org' in author:
            orgs.append(author['org'])
        else:
            orgs.append(None)

        has_author = data['authors'].apply(lambda x: author in x)
        citations.append(list(data[has_author]['citations']))

    data = pd.DataFrame({'name': names,
                        'org': orgs,
                        'citation_list': citations})

    def top(df):
        df['papers'] = df['citation_list'].apply(len)
        df['citations'] = df['citation_list'].apply(np.sum)
        df['citations_per_paper'] = df['citations']/df['papers']
        df['standard deviation'] = df['citation_list'].apply(np.std)
        df = df.drop('citation_list', axis=1)
        df = df.sort_values('citations_per_paper', ascending=False)
        df = df.head(30)
        return df

    author = data.copy()
    author = top(author)
    author.index = range(1, 31)
    author.to_csv('./results/top30/top authors.csv')

    org = data.copy()
    org['author_num'] = 1
    citation_list = org.groupby('org')['citation_list'].sum()
    org = org.groupby('org').sum()
    org['citation_list'] = citation_list
    org = top(org)
    org.to_csv('./results/top30/top orgs.csv')


if __name__ == '__main__':
    if not os.path.exists('./results/top30'):
        os.mkdir('./results/top30')

    jrnl_papers = pd.read_json('./cache/journal_paper_complete.txt')
    conf_papers = pd.read_json('./cache/conference_paper_complete.txt')

    top_papers(jrnl_papers, 'journal')
    top_papers(conf_papers, 'conference')

    papers = pd.concat([jrnl_papers, conf_papers])
    top_authors_and_orgs(papers)

    print(now(), 'All Done!')
