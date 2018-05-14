import json
import os
import time

import numpy as np
import pandas as pd

start_year, end_year = 1996, 2016
k = 30


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def top_papers(data, type_str):
    print(now(), 'finding top papers')
    
    cond = data['year'].isin(range(start_year, end_year))
    data = data[cond].copy()

    data = data.sort_values('citations', ascending=False)
    top = data.head(k)[['title', 'citations', 'year']]
    top.index = range(1, k+1)
    top.to_csv('./results/top30/top {} papers.csv'.format(type_str))


def keep_it_real(x):
    if pd.isna(x):
        return np.nan
    x = [word for word in x.split('#TAB#') if word!='']
    x = ' '.join(x)
    x = x.split('|||')[0]
    # x = x.split('|')[-1]
    return x.split(',')[0]


def top_authors_and_orgs(data):
    print(now(), 'finding top authors and orgs')

    cond = data['year'].isin(range(start_year, end_year))
    data = data[cond].copy()

    name = []
    org = []
    citations = []
    for index, row in data.iterrows():
        if 'authors' in row:
            for author in row['authors']:
                name.append(author['name'])
                citations.append(row['citations'])
                if 'org' in author:
                    org.append(author['org'])
                else:
                    org.append(np.nan)
    data = pd.DataFrame({'name': name,
                         'org': org,
                         'citations': citations})
    data['papers'] = 1
    data['org'] = data['org'].apply(keep_it_real)

    authors = data.groupby(['name', 'org'], as_index=False).sum()
    authors['citations_per_paper'] = authors['citations']/authors['papers']
    authors = authors.sort_values('citations_per_paper', ascending=False)
    authors.index = range(1, len(authors)+1)
    top_authors = authors
    top_authors.to_csv('./results/top30/top authors.csv')
    
    authors['author_num'] = 1
    orgs = authors.groupby('org', as_index=False).sum()
    orgs['citations_per_paper'] = orgs['citations']/orgs['papers']
    orgs = orgs.sort_values('citations_per_paper', ascending=False)
    orgs.index = range(1, len(orgs)+1)
    top_orgs = orgs.head(k)
    orgs.to_csv('./results/top30/top orgs.csv')


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
