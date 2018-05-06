import json
import os
import time

import numpy as np
import pandas as pd


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


FILE_NUM = 166
DATA_PATH = 'D:\\Dataset\\mag\\mag_papers_{}.txt'

titles = {}
with open('./titles.txt', 'r') as f:
    for line in f:
        t, v, typ = line.split('\t')
        titles[t.strip().lower()] = [v, typ]
        
data = []
for i in range(FILE_NUM):
    print(now(), i)

    with open(DATA_PATH.format(i), 'r', encoding='UTF-8') as f:
        for line in f:
            l = json.loads(line)
            if 'title' not in l:
                continue
            t = l['title'].strip().lower()
            if t in titles:
                l['file'] = i
                l['real_venue'] = titles[t][0]
                l['type'] = titles[t][1]
                data.append(l)

        print(len(data))
        if len(data)==len(titles):
            break
            
data = pd.DataFrame(data)
data = data[['title', 'venue', 'real_venue', 'type', 'file']]
data.to_csv('./cache/venue_papers.csv')
