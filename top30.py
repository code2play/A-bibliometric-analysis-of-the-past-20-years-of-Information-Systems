import json
import os
import re
import time

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

start_year, end_year = 1996, 2016
k = 30
aff = pd.read_csv('./cache/Affiliations.txt', 
                  sep='\t', 
                  index_col=0, 
                  names=['name'])
aff = list(aff['name'])


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
    x = x.lower()

    # if 'stanford university' in x:
    #     return 'Stanford University'
    # if 'massachusetts institute of technology' in x or \
    #     x.startswith('mit'):
    #     return 'Massachusetts Institute of Technology'
    # if x.startswith('ibm'):
    #     return 'IBM'
    # if x.startswith('microsoft'):
    #     return 'Microsoft'
    x = x.replace('univ.', 'university')
    x = x.replace('unviersity', 'university')
    x = x.replace('dept.', 'dept')
    x = x.replace(' lab.', ' lab')
    x = x.replace(' inc.', ' inc')
    x = x.replace(' comp.', ' computer')
    x = x.replace(' sci.', ' science')
    x = x.replace('&', 'and')
    x = ' '.join([word.strip() for word in x.split('#tab#') if word!=''])
    x = x.split('|||')[0]
    x = x.split(',')[0]
    x = ' '.join([word.strip() for word in x.split('|') if word!=''])
    x = ' '.join([word.strip() for word in x.split('/') if word!=''])

    return x.title()


def match(x):
    x = x.lower()
    r = [fuzz.partial_ratio(x, a)+fuzz.ratio(x, a) for a in aff]
    index = np.argmax(r)
    if r[index]>150:
        x = aff[index]
    return x.title()


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
    authors['org'] = authors[authors['citations_per_paper']>100]['org'].apply(match)
    authors = authors.sort_values('citations_per_paper', ascending=False)
    authors.index = range(1, len(authors)+1)
    top_authors = authors
    top_authors.to_csv('./results/top30/top authors.csv')
    
    authors['author_num'] = 1
    orgs = authors.groupby('org', as_index=False).sum()
    orgs['citations_per_paper'] = orgs['citations']/orgs['papers']
    # orgs = orgs[orgs['papers']>1]
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

    papers = pd.concat([jrnl_papers, conf_papers], ignore_index=True)
    top_authors_and_orgs(papers)

    print(now(), 'All Done!')
