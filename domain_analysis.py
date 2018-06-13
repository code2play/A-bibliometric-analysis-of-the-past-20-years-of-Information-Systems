import json
import os
import time

import numpy as np
import pandas as pd

from top30 import keep_it_real

# FILE_NUM = 167
# DATA_PATH = 'D:\\Dataset\\mag\\mag_papers_{}.txt'
FILE_NUM = 1
DATA_PATH = 'mag_papers_{}.txt'
start_year, end_year = 1996, 2016


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
def count_citation(data, type_str):
    paper_citation = dict.fromkeys(data.index, 0)
    paper_author_self_citation = dict.fromkeys(data.index, 0)
    paper_venue_self_citation = dict.fromkeys(data.index, 0)

    years = range(start_year, end_year)
    year_citation = dict.fromkeys(years, 0)
    year_author_self_citation = dict.fromkeys(years, 0)
    year_venue_self_citation = dict.fromkeys(years, 0)
    ref_interval = dict.fromkeys(years, 0)

    paper_citation_every_year = np.zeros((len(data), len(years)))
    paper_citation_every_year = pd.DataFrame(
        paper_citation_every_year,
        index=data.index,
        columns=years
    )

    for i in range(FILE_NUM):
        print(now(), i)
        with open(DATA_PATH.format(i), 'r', encoding='UTF-8') as f:
            for line in f:
                paper = json.loads(line)
                if 'year' not in paper:
                    continue
                year = paper['year']
                if year not in years:
                    continue
                if 'references' in paper:
                    for ref in paper['references']:
                        if ref in paper_citation:
                            paper_citation[ref] += 1
                            year_citation[year] += 1
                            ref_interval[year] += year-data.loc[ref, 'year']
                            paper_citation_every_year.loc[ref, year] += 1
                            # 判断自引
                            if ('authors' in paper) and (data.loc[ref, 'authors']!=None):
                                names1 = [a['name'] for a in paper['authors']]
                                names2 = [a['name'] for a in data.loc[ref, 'authors']]
                                for name in names1:
                                    if name in names2:
                                        paper_author_self_citation[ref] += 1
                                        year_author_self_citation[year] += 1
                                        break
                            if 'venue' in paper:
                                if paper['venue']==data.loc[ref, 'venue']:
                                    paper_venue_self_citation[ref] += 1
                                    year_venue_self_citation[year] += 1

    paper_citation = pd.DataFrame({'citations':paper_citation,
                                   'author_self_citations':paper_author_self_citation,
                                   'venue_self_citations':paper_venue_self_citation})
    year_citation = pd.DataFrame({'citations':year_citation,
                                  'author_self_citations':year_author_self_citation,
                                  'venue_self_citations':year_venue_self_citation,
                                  'ref_interval_sum':ref_interval})
    year_citation.to_csv('./cache/{}_citation_annual_summary.csv'.format(type_str))
    paper_citation_every_year.to_csv('./cache/{}_citation_every_year.csv'.format(type_str))
    return paper_citation, year_citation, paper_citation_every_year


def get_citation_info(data, type_str):
    print('{} Start getting {} citation data'.format(now(), type_str))

    filename1 = './cache/{}_paper_complete.txt'.format(type_str)
    filename2 = './cache/{}_citation_annual_summary.csv'.format(type_str)
    filename3 = './cache/{}_citation_every_year.csv'.format(type_str)
    if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3):
        data = pd.read_json(filename1, typ='frame')
        year_citation = pd.read_csv(filename2, index_col=0)
        paper_citation_every_year = pd.read_csv(filename3, index_col=0)
        return data, year_citation, paper_citation_every_year

    cols = ['year', 'authors', 'venue']
    temp = data[cols]
    temp.index = data['id']
    paper_citation, year_citation, paper_citation_every_year = count_citation(temp, type_str)

    data = data.merge(paper_citation, left_on='id', right_index=True, how='left')
    data.to_json(filename)

    return data, year_citation, paper_citation_every_year


def num_of_uniq_authors(authors):
    authors = [a for index, row in authors.iteritems() if row!=None for a in row]
    authors = pd.DataFrame(authors)
    if len(authors)==0:
        return 0
    # if 'org' in authors.columns:
    #     authors['org'] = authors['org'].apply(keep_it_real)
    #     authors = authors.groupby(['name', 'org'])
    # else:
    authors = authors.groupby(['name'])
    return len(authors)


def basic_info(data, type_str):
    print('{} Start concluding {} basic infomation'.format(now(), type_str))
    if not os.path.exists('./results'):
        os.mkdir('./results')

    cond = data['year'].isin(range(start_year, end_year))
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
    data = data.groupby(['real_venue'], as_index=False, sort=True).sum()
    data = data.merge(year_df, on='real_venue')

    data['paper_per_author'] = data['papers_num']/data['uniq_authors']
    data['author_per_paper'] = data['authors_num']/data['papers_num']
    data['citation_per_paper'] = data['citations']/data['papers_num']
    data = data.drop(['year'], axis=1)
    data.to_csv('./results/{} basic info.csv'.format(type_str))


def annual_summary(data, year_citation, paper_citation_every_year, type_str):
    print('{} Start calculating annual summary for {} papers'.format(now(), type_str))
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/growth of domain & impact'):
        os.mkdir('./results/growth of domain & impact')

    cond = data['year'].isin(range(start_year, end_year))
    data = data[cond].copy()

    authors = data['authors'][data['authors'].notna()]
    data['authors_num'] = authors.apply(len)

    refs = data['references'][data['references'].notna()]
    data['reference_num'] = refs.apply(len)

    data['papers_num'] = 1
    uniq_authors = [num_of_uniq_authors(data[data['year']==y]['authors']) \
                    for y in range(start_year, end_year)]
    data = data.drop(['citations', 'author_self_citations', 'venue_self_citations'], axis=1)
    data = data.groupby(['year'], as_index=False, sort=True).sum()
    data['uniq_authors'] = uniq_authors

    citations = np.asarray(paper_citation_every_year.sum())
    paper_num = np.asarray((paper_citation_every_year>0).sum())
    year_citation['citation_per_paper'] = citations/paper_num

    data = data.merge(year_citation, left_on='year', right_index=True)

    data['paper_per_author'] = data['papers_num']/data['uniq_authors']
    data['author_per_paper'] = data['authors_num']/data['papers_num']
    data['reference_per_paper'] = data['reference_num']/data['papers_num']
    data['author_self_citation_rate'] = data['author_self_citations']/data['citations']
    data['venue_self_citation_rate'] = data['venue_self_citations']/data['citations']
    data['mean_ref_interval'] = data['ref_interval_sum']/data['citations']
    data = data.drop(['ref_interval_sum'], axis=1)

    g1 = data.iloc[1:, :]
    g2 = data.iloc[:-1,:]
    g1.index = g2.index = range(len(g1))
    cols = ['citations', 'authors_num', 'papers_num', 'paper_per_author']
    growth_rate = (g1.loc[:,cols]-g2.loc[:,cols])/g2.loc[:,cols]
    cols = [item+'_growth_rate' for item in cols]
    growth_rate.columns = cols
    growth_rate['year'] = g1['year']

    res = data.merge(growth_rate, on='year', how='left')
    res.index = range(1, len(res)+1)

    res.to_csv('./results/growth of domain & impact/{} annual summary.csv'.format(type_str, type_str))


if __name__ == '__main__':
    jrnl_paper, conf_paper = select_data()

    jrnl_paper, jrnl_year_citation, jrnl_citation_every_year = get_citation_info(jrnl_paper, 'journal')
    conf_paper, conf_year_citation, conf_citation_every_year = get_citation_info(conf_paper, 'conference')

    cols = ['abstract', 'doc_type', 'doi', 'fos', 'issue', 'keywords', 'lang',
            'n_citation', 'page_end', 'page_start', 'publisher', 'title', 'url', 'volume']
    jrnl_paper = jrnl_paper.drop(cols, axis=1)
    conf_paper = conf_paper.drop(cols, axis=1)

    basic_info(jrnl_paper, 'journal')
    annual_summary(jrnl_paper, jrnl_year_citation, jrnl_citation_every_year, 'journal')

    basic_info(conf_paper, 'conference')
    annual_summary(conf_paper, conf_year_citation, conf_citation_every_year, 'conference')

    all_paper = pd.concat([jrnl_paper, conf_paper], ignore_index=True)
    all_year_citation = jrnl_year_citation + conf_year_citation
    all_citation_every_year = pd.concat([jrnl_citation_every_year, conf_citation_every_year])
    annual_summary(all_paper, all_year_citation, all_citation_every_year, 'all')

    print(now(), 'All Done!')
