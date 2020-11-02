#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re

with open("ci.rhythmic.json","r",encoding="utf-8") as file:
    rhythmics = json.load(file)

with open("ci.rhythmic.link.json","r",encoding="utf-8") as file:
    links = json.load(file)

roots = {}
cnt = {}

def find(r):
    if roots[r] != r:
        roots[r] = find(roots[r])
    return roots[r]

def Union(x, y):
    xx = find(x)
    yy = find(y)
    if xx == yy:
        return
    if cnt[xx] <= cnt[yy]:
        roots[xx] = yy
    else:
        roots[yy] = xx
    return

for rhythmic in rhythmics:
    r = rhythmic['rhythmic']
    cnt[r] = cnt[r] + 1 if r in cnt else 1

for r in cnt:
    roots[r] = r

for link in links:
    for r in link[:-1]:
        Union(r, link[-1])

for r in cnt:
    if r != find(r):
        #print("词牌合并:", r, cnt[r], find(r), cnt[find(r)], 'to', cnt[find(r)]+cnt[r])
        cnt[find(r)] += cnt[r]

for rhythmic in rhythmics:
    rhythmic['rhythmic'] = find(rhythmic['rhythmic'])


with open("ci.rhythmic.root.json","w",encoding="utf-8") as file:
    json.dump(roots, file, indent = 2, ensure_ascii=False)

with open("ci.rhythmic.cnt.json","w",encoding="utf-8") as file:
    json.dump(cnt, file, indent = 2, ensure_ascii=False)

with open("ci.rhythmic.all.json","w",encoding="utf-8") as file:
    json.dump(rhythmics, file, indent = 2, ensure_ascii=False)

print('完成词牌合并（合并细节出于log简洁已注释掉）')

sorted_cnt = sorted(cnt.items(), key=lambda item:item[1], reverse = True)
print(sorted_cnt[:10])
