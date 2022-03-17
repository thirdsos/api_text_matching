
from itertools import count
from re import A
import numpy as np
import pandas as pd
from collections import defaultdict



df = pd.read_csv('keyword_new.csv', sep='|')
# print(df.head())
word = df['word'].apply(lambda s: s.split('/sp/')).tolist()
keyword_pages_idx = defaultdict(list)
tmp = []
for i, wd in enumerate(word):
    for w in wd:
        keyword_pages_idx[w].append(i)
        tmp.append(w)
word = np.unique(tmp)
word = pd.Series(word, name='unique')
word.to_csv('unique_keyword_new.csv', index=False)

# import json
# with open('unique_keyword_idx.json', 'w', encoding='utf-8') as file:
    # json.dump(keyword_pages_idx, file)
tex =[]
with open('unique_keyword_idx_new.txt', 'w', encoding='utf-8') as file:
    dic = str(dict(keyword_pages_idx))
    file.write(dic)
keyword_pages_idx = eval(dic)

# print(keyword_pages_idx['ล้านบาท'])

