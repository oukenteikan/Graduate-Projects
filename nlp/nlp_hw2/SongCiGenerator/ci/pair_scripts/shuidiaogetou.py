#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re
import random

with open("ci.master.train.json","r",encoding="utf-8") as file:
    train_songci = json.load(file)

with open('ci.master.test.json', 'r', encoding="utf-8")  as file:
    test_songci = json.load(file)

with open("../rhythmic_scrips/ci.rhythmic.root.json","r",encoding="utf-8") as file:
    roots = json.load(file)

train_sdgt_pairs = []

def is_valid(sentence):
    for s in sentence:
        if s < '\u4E00' or s > '\u9FEF':
            if s != '\u25a1':
                print(s, s.encode('unicode-escape'), sentence)
            return False
    if len(sentence) == 0:
        return False
    return True

for ci in train_songci:
    rhythmics = ci['rhythmic'].split('・')
    rhythmics = list(filter(lambda x: is_valid(x), rhythmics))
    if len(rhythmics) == 0:
        continue
    root = roots[rhythmics[0]]
    if root == '水调歌头':
        p = ci['paragraphs']
        for i in range(len(p)-1):
            src = p[i]
            trg = p[i+1]
            if is_valid(src) and is_valid(trg):
                pair = {"src":src, "trg":trg}
                train_sdgt_pairs.append(pair)

pair_lists = {}

for i in range(2, 9):
    for j in range(2, 9):
        pair_lists[(i, j)] = []

for p in train_sdgt_pairs:
    l1 = len(p['src'])
    l2 = len(p['trg'])
    if (l1, l2) in pair_lists:
        pair_lists[(l1, l2)].append(p)

for i in range(2, 9):
    for j in range(2, 9):
        with open('train_sdgt_pairs/{}_{}.json'.format(i, j), 'w', encoding="utf-8")  as fout:
            json.dump(pair_lists[(i, j)], fout, indent=2, ensure_ascii=False)

test_data = []

for ci in test_songci:
    rhythmics = ci['rhythmic'].split('・')
    rhythmics = list(filter(lambda x: is_valid(x), rhythmics))
    if len(rhythmics) == 0:
        continue
    root = roots[rhythmics[0]]
    if root == '水调歌头':
        src = ci['paragraphs'][0]
        trg = ''.join(ci['paragraphs'][1:])
        test_data.append({'src':src, 'trg':trg, 'rhythmic':root})

with open('ci.test.sdgt.json', 'w', encoding="utf-8")  as fout:
    json.dump(test_data, fout, indent=2, ensure_ascii=False)

