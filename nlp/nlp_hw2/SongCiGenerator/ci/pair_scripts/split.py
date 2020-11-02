#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re

with open("ci.all.json","r",encoding="utf-8") as file:
    songci = json.load(file)

def saturate(sentence):
    sentence = sentence.replace('…', '')
    sentence = sentence.replace('ピ', '')
    sentence = sentence.replace('け', '翛')
    sentence = sentence.replace('シ', '蘼')
    return sentence

for ci in songci:
    sentences = []
    for s in ci['paragraphs']:
        s = re.split('，|。|！|？|、', s)
        for ss in s:
            sss = saturate(ss)
            if len(sss) != 0:
                sentences.append(sss)
    ci['paragraphs'] = sentences

with open("ci.all.split.json","w",encoding="utf-8") as file:
    json.dump(songci, file, indent = 2, ensure_ascii=False)

