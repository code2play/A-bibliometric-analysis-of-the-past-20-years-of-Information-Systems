import json
import os
import time

import numpy as np
import pandas as pd

from top30 import keep_it_real

FILE_NUM = 1
# DATA_PATH = 'D:\\Dataset\\mag\\mag_papers_{}.txt'
DATA_PATH = 'mag_papers_{}.txt'
start_year, end_year = 1995, 2016


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())

# 选出指定期刊或会议的论文
def select_data():
    print('{} Start selecting data'.format(now()))
    if not os.path.exists('./cache'):
        os.mkdir('cache')

    jrnl_filename = './cache/journal_papers.txt'
    conf_filename = './cache/conference_papers.txt'

    if os.path.exists(jrnl_filename) and os.path.exists(conf_filename):
        jrnl_paper = pd.read_json(jrnl_filename, typ='frame')
        conf_paper = pd.read_json(conf_filename, typ='frame')
        return jrnl_paper, conf_paper

    jrnl = {}   # 期刊名
    conf = {}   # 会议名
    with open('./journals.txt', 'r') as f:
        for line in f:
            name, real_name = line.split('\t')
            jrnl[name.strip().lower()] = real_name.strip().lower()
    with open('./conferences.txt', 'r') as f:
        for line in f:
            name, real_name = line.split('\t')
            conf[name.strip().lower()] = real_name.strip().lower()

    jrnl_paper = []
    conf_paper = []
    for i in range(FILE_NUM):
        print(now(), i)
        with open(DATA_PATH.format(i), 'r', encoding='UTF-8') as f:
            for line in f:
                data = json.loads(line)
                if 'venue' not in data:
                    continue
                venue = data['venue'].strip().lower()
                if venue in jrnl:
                    data['real_venue'] = jrnl[venue]
                    jrnl_paper.append(data)
                elif venue in conf:
                    data['real_venue'] = conf[venue]
                    conf_paper.append(data)
    jrnl_paper = pd.DataFrame(jrnl_paper)
    conf_paper = pd.DataFrame(conf_paper)
    jrnl_paper.index = range(len(jrnl_paper))
    conf_paper.index = range(len(conf_paper))
    jrnl_paper.to_json(jrnl_filename)
    conf_paper.to_json(conf_filename)
    return jrnl_paper, conf_paper

# 统计引用信息
def count_citation(paper_ids, authors, ref_ids, type_str):
    print('{} Start counting {} citation'.format(now(), type_str))

    citation_filename = './cache/{}_paper_citations.txt'.format(type_str)
    papers_filename = './cache/{}_cited_papers.csv'.format(type_str)
    if os.path.exists(citation_filename) and os.path.exists(papers_filename):
        citations = pd.read_json(citation_filename, typ='frame')
        cited_papers = pd.read_csv(papers_filename, index_col=0)
        return citations, cited_papers

    citation = dict.fromkeys(paper_ids, 0)      # 引用数
    self_citation = dict.fromkeys(paper_ids, 0) # 自引数
    cited_papers = dict.fromkeys(ref_ids)       # 被期刊或会议中论文引用的论文的发表时间，用于计算引文时间差
    for i in range(FILE_NUM):
        print(now(), i)
        with open(DATA_PATH.format(i), 'r', encoding='UTF-8') as f:
            for line in f:
                data = json.loads(line)
                if data['id'] in cited_papers:
                    cited_papers[data['id']] = data['year']
                if 'references' not in data:
                    continue
                for ref in data['references']:
                    if ref not in citation:
                        continue
                    citation[ref] += 1
                    # 判断自引
                    if 'authors' not in data:
                        continue
                    for author in data['authors']:
                        if author in authors[ref]:
                            self_citation[ref] += 1
                            break
    citations = pd.DataFrame({'citations':citation, 'self_citations':self_citation})
    citations.to_json(citation_filename)
    cited_papers = pd.DataFrame({'year': cited_papers})
    cited_papers.to_csv(papers_filename)
    return citations, cited_papers

# 合并数据集
def get_citation_info(data, type_str):
    print('{} Start getting {} citation data'.format(now(), type_str))

    filename = './cache/{}_paper_complete.txt'.format(type_str)
    if os.path.exists(filename):
        data = pd.read_json(filename, typ='frame')
        return data

    paper_ids = data['id'].values
    authors = data['authors']
    authors.index = data['id']
    # 所有引文id
    ref_ids = [j for i in data[data['references'].notna()]['references'] for j in i]

    citations, cited_papers = count_citation(paper_ids, authors, ref_ids, type_str)

    # 引文时间差
    def interval(x):
        if x==None:
            return None
        b = cited_papers.loc[x].min().values[0]
        e = cited_papers.loc[x].max().values[0]
        return e-b
    data['ref_interval'] = data['references'].apply(interval)

    data = data.merge(citations, left_on='id', right_index=True, how='left')
    data.to_json(filename)
    return data


def num_of_uniq_authors(authors):
    authors = [a for index, row in authors.iteritems() if row!=None for a in row]
    authors = pd.DataFrame(authors)
    if len(authors)==0:
        return 0
    if 'org' in authors.columns:
        authors['org'] = authors['org'].apply(keep_it_real)
        authors = authors.groupby(['name', 'org'])
    else:
        authors = authors.groupby(['name'])
    return len(authors)


def basic_info(data, type_str):
    print('{} Start concluding {} basic infomation'.format(now(), type_str))
    if not os.path.exists('./results'):
        os.mkdir('./results')

    cond = data['year'].isin(range(start_year+1, end_year))
    data = data[cond].copy()

    authors = data['authors'][data['authors'].notna()]
    data['authors_num'] = authors.apply(len)

    venues = data['real_venue'].unique()
    years = []
    sy = []
    ey = []
    freq = []
    uniq_authors = []
    for venue in venues:
        cond = (data['real_venue']==venue)
        uniq_years = data[cond]['year'].unique()
        years.append(sorted(uniq_years))
        s = uniq_years.min()
        e = uniq_years.max()
        sy.append(s)
        ey.append(e)
        freq.append((e-s+1)/len(uniq_years))
        uniq_authors.append(num_of_uniq_authors(data[cond]['authors']))

    year_df = pd.DataFrame({'real_venue':venues,
                            'start_year':sy,
                            'end_year':ey,
                            'freq':freq,
                            'uniq_authors':uniq_authors,
                            'years':years})
    year_df['num_years'] = year_df['years'].apply(len)

    data['papers_num'] = 1
    data = data.drop(['issue', 'n_citation', 'page_end', 'page_start', 'volume'], axis=1)
    data = data.groupby(['real_venue'], as_index=False, sort=True).sum()
    data = data.merge(year_df, on='real_venue')

    data['paper_per_author'] = data['papers_num']/data['authors_num']
    data['author_per_paper'] = data['uniq_authors']/data['papers_num']
    data['citation_per_paper'] = data['citations']/data['papers_num']
    data = data.drop(['ref_interval', 'self_citations', 'year'], axis=1)
    data.to_csv('./results/{} basic info.csv'.format(type_str))


def annual_summary(data, type_str, ALL=True):
    if ALL==True:
        print('{} Start calculating annual summary for all {} papers'.format(now(), type_str))
    else:
        print('{} Start calculating annual summary for each {} venues'.format(now(), type_str))
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/growth of domain & impact'):
        os.mkdir('./results/growth of domain & impact')
    if not os.path.exists('./results/growth of domain & impact/{}'.format(type_str)):
        os.mkdir('./results/growth of domain & impact/{}'.format(type_str))

    cond = data['year'].isin(range(start_year, end_year))
    data = data[cond].copy()

    authors = data['authors'][data['authors'].notna()]
    data['authors_num'] = authors.apply(len)

    refs = data['references'][data['references'].notna()]
    data['reference_num'] = refs.apply(len)

    data['papers_num'] = 1
    # data['key_papers'] = data['citations']>100
    data = data.drop(['issue', 'n_citation', 'page_end', 'page_start', 'volume'], axis=1)

    if ALL==True:
        uniq_authors = [num_of_uniq_authors(data[data['year']==y]['authors']) \
                        for y in range(start_year, end_year)]
        data = data.groupby(['year'], as_index=False, sort=True).sum()
        data['uniq_authors'] = uniq_authors
    else:
        venues = sorted(data['real_venue'].unique())
        uniq_authors = []
        for y in range(start_year, end_year):
            for v in venues:
                cond = (data['year']==y) & (data['real_venue']==v)
                num = num_of_uniq_authors(data[cond]['authors'])
                if num!=0:
                    uniq_authors.append(num)
        data = data.groupby(['year', 'real_venue'], as_index=False, sort=True).sum()
        data['uniq_authors'] = uniq_authors

    data['paper_per_author'] = data['papers_num']/data['authors_num']
    data['author_per_paper'] = data['uniq_authors']/data['papers_num']
    data['citation_per_paper'] = data['citations']/data['papers_num']
    data['reference_per_paper'] = data['reference_num']/data['papers_num']
    data['self_citation_rate'] = data['self_citations']/data['citations']
    data['mean_ref_interval'] = data['ref_interval']/data['papers_num']
    # data['key_papers_rate'] = data['key_papers']/data['papers_num']
    data = data.drop(['reference_num', 'ref_interval'], axis=1)

    def growth_rate(df):
        g1 = df.iloc[1:, :]
        g2 = df.iloc[:-1,:]
        g1.index = g2.index = range(len(g1))
        cols = ['citations', 'authors_num', 'papers_num', 'paper_per_author']
        growth_rate = (g1.loc[:,cols]-g2.loc[:,cols])/g2.loc[:,cols]
        cols = [item+'_growth_rate' for item in cols]
        growth_rate.columns = cols
        growth_rate['year'] = g1['year']

        res = g1.merge(growth_rate, on='year', how='left')
        res.index = range(1, len(res)+1)
        return res

    if ALL==True:
        res = growth_rate(data)
        res.to_csv('./results/growth of domain & impact/{}/all {} annual summary.csv'.format(type_str, type_str))
    else:
        venues = data['real_venue'].unique()
        for venue in venues:
            cond = (data['real_venue']==venue)
            res = growth_rate(data[cond])
            res.to_csv('./results/growth of domain & impact/{}/{} annual summary.csv'.format(type_str, venue))


if __name__ == '__main__':
    jrnl_paper, conf_paper = select_data()

    jrnl_paper = get_citation_info(jrnl_paper, 'journal')
    basic_info(jrnl_paper, 'journal')
    annual_summary(jrnl_paper, 'journal', True)
    annual_summary(jrnl_paper, 'journal', False)

    conf_paper = get_citation_info(conf_paper, 'conference')
    basic_info(conf_paper, 'conference')
    annual_summary(conf_paper, 'conference', True)
    annual_summary(conf_paper, 'conference', False)

    all_paper = pd.concat([jrnl_paper, conf_paper], ignore_index=True)
    annual_summary(all_paper, 'all', True)

    print(now(), 'All Done!')
