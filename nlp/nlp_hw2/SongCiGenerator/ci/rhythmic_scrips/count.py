#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re

with open("ci.rhythmic.all.json","r",encoding="utf-8") as file:
    rhythmics = json.load(file)

with open("ci.rhythmic.root.json","r",encoding="utf-8") as file:
    roots = json.load(file)

with open("ci.rhythmic.cnt.json","r",encoding="utf-8") as file:
    cnt = json.load(file)

def cmp(x):
    return x['cnt']

for r in roots:
    if r != roots[r]:
        continue
    subset = []
    for rhythmic in rhythmics:
        if rhythmic['rhythmic'] == r:
            subset.append(rhythmic)
    with open("rhythmics/{}.json".format(r),"w",encoding="utf-8") as file:
        json.dump(subset, file, indent = 2, ensure_ascii=False)

    for i in range(len(subset)):
        subset[i]['cnt'] = 0
        for j in range(len(subset)):
            if subset[i]['num'] == subset[j]['num'] \
                    and subset[i]['lengths'] == subset[j]['lengths'] \
                    and subset[i]['punctuations'] == subset[j]['punctuations']:
                        subset[i]['cnt'] += 1
    uniqueset = []
    for t in subset:
        if t not in uniqueset:
            uniqueset.append(t)

    sortedset = sorted(uniqueset, key = cmp, reverse = True)
    for t in sortedset:
        t['prob'] = t['cnt'] / float(cnt[t['rhythmic']])

    distribute = [sortedset[i]['prob'] for i in range(len(sortedset))]

    index = 1
    while sum(distribute[:index]) <= 0.5: index += 1
    if index > 10:
        print(r, index, '/', len(distribute))

    with open("rhythmics/{}.sort.json".format(r),"w",encoding="utf-8") as file:
        json.dump(sortedset, file, indent = 2, ensure_ascii=False)
    
    with open("rhythmics/{}.main.json".format(r),"w",encoding="utf-8") as file:
        json.dump(sortedset[:index], file, indent = 2, ensure_ascii=False)

print('按词牌分类输出到rhythmics/*json')
print('去重并从大到小排序输出到rhythmics/*sort.json')
print('选取几种主要格式使其能够占到半数以上，print主要格式很多的词牌，最终结果输出到rhythmics/*main.json')
