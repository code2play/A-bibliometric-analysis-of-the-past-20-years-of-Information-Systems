import json
import os
import time

import numpy as np
import pandas as pd


FILE_NUM = 1
# DATA_PATH = 'D:\\Dataset\\mag\\mag_papers_{}.txt'
DATA_PATH = 'mag_papers_{}.txt'
start_year, end_year = 1995, 2017


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
    print('{} Start counting citation'.format(now()))

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
    print('{} Start getting citation data'.format(now()))

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
        b = cited_papers.loc[x].min().values[0]
        e = cited_papers.loc[x].max().values[0]
        return e-b
    data['ref_interval'] = data[data['references'].notna()]['references'].apply(interval)

    data = data.merge(citations, left_on='id', right_index=True, how='left')
    data.to_json(filename)
    return data


def annual_summary(data, type_str):
    print('{} Start calculating annual summary'.format(now()))
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/{}'.format(type_str)):
        os.mkdir('./results/{}'.format(type_str))

    authors = data['authors'][data['authors'].notna()]
    data['authors_num'] = authors.apply(len)

    refs = data['references'][data['references'].notna()]
    data['reference_num'] = refs.apply(len)

    data['papers_num'] = 1
    data['key_papers'] = data['citations']>100
    data = data.drop(['issue', 'n_citation', 'page_end', 'page_start', 'volume'], axis=1)
    # data = data.groupby(['year', 'real_venue'], as_index=False, sort=True).sum()
    data = data.groupby(['year'], as_index=False, sort=True).sum()

    data['paper_per_person'] = data['papers_num']/data['authors_num']
    data['citation_per_paper'] = data['citations']/data['papers_num']
    data['reference_per_paper'] = data['reference_num']/data['papers_num']
    data['self_citation_rate'] = data['self_citations']/data['citations']
    data['mean_ref_interval'] = data['ref_interval']/data['papers_num']
    data['key_papers_rate'] = data['key_papers']/data['papers_num']
    data = data.drop(['reference_num', 'ref_interval'], axis=1)

    # 增长率
    # venues = data['real_venue'].unique()
    # for venue in venues:
        # cond = (data['real_venue']==venue) & (data['year'].isin(range(start_year, end_year)))
    cond = data['year'].isin(range(start_year, end_year))
    g1 = data[cond].iloc[1:, :]
    g2 = data[cond].iloc[:-1,:]
    g1.index = g2.index = range(len(g1))
    cols = ['citations', 'authors_num', 'papers_num', 'paper_per_person', 'key_papers_rate']
    growth_rate = (g1.loc[:,cols]-g2.loc[:,cols])/g2.loc[:,cols]
    cols = [item+'_growth_rate' for item in cols]
    growth_rate.columns = cols
    growth_rate['year'] = g1['year']

    res = g1.merge(growth_rate, on='year', how='left')
    # res.to_csv('./results/{}/{} annual summary.csv'.format(type_str, venue))
    res.to_csv('./results/{}/all {} annual summary.csv'.format(type_str, type_str))

    return data


if __name__ == '__main__':
    jrnl_paper, conf_paper = select_data()
    jrnl_paper = get_citation_info(jrnl_paper, 'journal')
    annual_summary(jrnl_paper, 'journal')
    conf_paper = get_citation_info(conf_paper, 'conference')
    annual_summary(conf_paper, 'conference')
