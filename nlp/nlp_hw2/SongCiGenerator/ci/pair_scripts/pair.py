#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re

with open("ci.all.split.json","r",encoding="utf-8") as file:
    songci = json.load(file)

def is_valid(sentence):
    for s in sentence:
        if s < '\u4E00' or s > '\u9FEF':
            if s != '\u25a1':
            #if True:
               	print(s, s.encode('unicode-escape'), sentence)
            return False
    if len(sentence) == 0:
        return False
    return True

pairs = []

for ci in songci:
    p = ci['paragraphs']
    for i in range(len(p)-1):
        src = p[i]
        trg = p[i+1]
        if is_valid(src) and is_valid(trg):
            pair = {"src":src, "trg":trg}
            pairs.append(pair)

with open('ci.pair.json', 'w', encoding="utf-8")  as fout:
    json.dump(pairs, fout, indent=2, ensure_ascii=False)
