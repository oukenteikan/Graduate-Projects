#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import random
import re
import jieba

BALANCE = 8192

with open("ci.pair.json","r",encoding="utf-8") as file:
    pairs = json.load(file)

def count(pairs):
    cnt = {}
    for p in pairs:
        l1 = len(p['src'])
        l2 = len(p['trg'])
        #if l1 > 8 or l2 > 8:
        #    print(p)
        #if l1 < 2 or l1 < 2:
        #    print(p)
        cnt[(l1, l2)] = cnt[(l1, l2)] + 1 if (l1, l2) in cnt else 1
    print(len(pairs))
    print(cnt)
    return len(pairs), cnt

def is_valid(sentence):
    for s in sentence:
        if s < '\u4E00' or s > '\u9FEF':
            #if s != '\u25a1':
                #print(s, s.encode('unicode-escape'), sentence)
            return False
    if len(sentence) == 0:
        return False
    return True

pair_l, cnt = count(pairs)

with open("ci.all.split.json","r",encoding="utf-8") as file:
    songci = json.load(file)

songci = list(filter(lambda ci: is_valid("".join(ci['paragraphs'])), songci))

while min(cnt.values()) < BALANCE:
    index = random.randint(0, len(songci)-1)
    p = songci[index]['paragraphs']
    clips = []
    for pp in p:
        clips.extend(jieba.lcut(pp, cut_all=False, HMM=True))
    if len(clips) < 3:
        #print(ci, clips)
        continue
    lengths = []
    lengths.append(0)
    for i in range(len(clips)):
        lengths.append(lengths[i] + len(clips[i]))
    for i in range(0, len(lengths)):
        for j in range(i+1, len(lengths)):
            for k in range(j+1, len(lengths)):
                if random.random() < 0.1:
                    pair = (lengths[j] - lengths[i], lengths[k] - lengths[j])
                    if pair in cnt and cnt[pair] < BALANCE:
                        src = "".join(clips[i:j])
                        trg = "".join(clips[j:k])
                        p = {"src":src, "trg":trg}
                        pairs.append({"src":src, "trg":trg})
                        print(f"We only have {cnt[pair]} {pair}, so we make a fake pair {src} {trg}")
                        cnt[pair] += 1

all_pair_l, fake_cnt = count(pairs)

with open('ci.pair.fake.json', 'w', encoding="utf-8")  as fout:
    json.dump(pairs[pair_l:], fout, indent=2, ensure_ascii=False)
with open('ci.pair.all.json', 'w', encoding="utf-8")  as fout:
    json.dump(pairs, fout, indent=2, ensure_ascii=False)
