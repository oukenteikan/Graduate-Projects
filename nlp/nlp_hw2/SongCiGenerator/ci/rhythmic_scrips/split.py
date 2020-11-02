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

def is_valid(sentence):
    for s in sentence:
        if s < '\u4E00' or s > '\u9FEF':
            if s != '\u25a1':
                print(s, s.encode('unicode-escape'), sentence)
            return False
    if len(sentence) <= 0:
        return False
    return True

rawRhythmicDatabase = []
rhythmicLink = []
for ci in songci:
    paragraphs = u''.join(ci['paragraphs'])
    sentences = []
    punctuations = []
    lengths = []
    i = 0
    valid = True
    for j in range(len(paragraphs)):
        if paragraphs[j] in [u'，', u'。', u'！', u'？', u'、']:
            sentence = saturate(paragraphs[i:j])
            if '（' in sentence:
                valid = False
            sentences.append(sentence)
            lengths.append(len(sentence))
            punctuations.append(paragraphs[j])
            i = j + 1
    if not valid:
        continue
    rhythmics = ci['rhythmic'].split('・')
    valid = True
    for r in rhythmics:
        if is_valid(r):
            rhythmic = {'rhythmic':r, 'num':len(sentences), 'lengths':lengths, 'punctuations':punctuations}
            rawRhythmicDatabase.append(rhythmic)
        else:
            valid = False
    if len(rhythmics) > 1 and valid:
        rhythmicLink.append(rhythmics)

print(len(songci))
print(len(rawRhythmicDatabase))
print(len(rhythmicLink))

with open("ci.rhythmic.json","w",encoding="utf-8") as file:
    json.dump(rawRhythmicDatabase, file, indent = 2, ensure_ascii=False)

with open("ci.rhythmic.link.json","w",encoding="utf-8") as file:
    json.dump(rhythmicLink, file, indent = 2, ensure_ascii=False)

print('洗掉内容，只留词牌和格式')
