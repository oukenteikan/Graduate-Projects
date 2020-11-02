#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import re
import jieba
import random

BALANCE = 8192

with open("ci.all.split.json","r",encoding="utf-8") as file:
    songci = json.load(file)

random.shuffle(songci)

train_songci = songci[:int(len(songci)*0.9)]
test_songci = songci[int(len(songci)*0.9):]

def is_valid(sentence):
    for s in sentence:
        if s < '\u4E00' or s > '\u9FEF':
            if s != '\u25a1':
                print(s, s.encode('unicode-escape'), sentence)
            return False
    if len(sentence) == 0:
        return False
    return True

def count(pairs):
    cnt = {}
    for p in pairs:
        l1 = len(p['src'])
        l2 = len(p['trg'])
        cnt[(l1, l2)] = cnt[(l1, l2)] + 1 if (l1, l2) in cnt else 1
    print(len(pairs))
    print(cnt)
    return len(pairs), cnt

pairs = []

for ci in train_songci:
    p = ci['paragraphs']
    for i in range(len(p)-1):
        src = p[i]
        trg = p[i+1]
        if is_valid(src) and is_valid(trg):
            pair = {"src":src, "trg":trg}
            pairs.append(pair)

with open('ci.pair.train.json', 'w', encoding="utf-8")  as fout:
    json.dump(pairs, fout, indent=2, ensure_ascii=False)

pair_lists = {}

for i in range(2, 9):
    for j in range(2, 9):
        pair_lists[(i, j)] = []

for p in pairs:
    l1 = len(p['src'])
    l2 = len(p['trg'])
    if (l1, l2) in pair_lists:
        pair_lists[(l1, l2)].append(p)

for i in range(2, 9):
    for j in range(2, 9):
        with open('train_pairs/{}_{}.json'.format(i, j), 'w', encoding="utf-8")  as fout:
            json.dump(pair_lists[(i, j)], fout, indent=2, ensure_ascii=False)

pair_l, cnt = count(pairs)

train_songci = list(filter(lambda ci: is_valid("".join(ci['paragraphs'])), train_songci))
test_songci = list(filter(lambda ci: is_valid("".join(ci['paragraphs'])), test_songci))

while min(cnt.values()) < BALANCE:
    index = random.randint(0, len(train_songci)-1)
    p = train_songci[index]['paragraphs']
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

pair_lists = {}

for i in range(2, 9):
    for j in range(2, 9):
        pair_lists[(i, j)] = []

for p in pairs:
    l1 = len(p['src'])
    l2 = len(p['trg'])
    if (l1, l2) in pair_lists:
        pair_lists[(l1, l2)].append(p)

for i in range(2, 9):
    for j in range(2, 9):
        with open('withfake_pairs/{}_{}.json'.format(i, j), 'w', encoding="utf-8")  as fout:
            json.dump(pair_lists[(i, j)], fout, indent=2, ensure_ascii=False)

with open("../rhythmic_scrips/ci.rhythmic.root.json","r",encoding="utf-8") as file:
    roots = json.load(file)

with open('ci.master.train.json', 'w', encoding="utf-8")  as fout:
    json.dump(train_songci, fout, indent=2, ensure_ascii=False)

with open('ci.master.test.json', 'w', encoding="utf-8")  as fout:
    json.dump(train_songci, fout, indent=2, ensure_ascii=False)

test_data = []

for ci in test_songci:
    if len(ci['paragraphs']) < 2:
        continue
    rhythmics = ci['rhythmic'].split('・')
    rhythmics = list(filter(lambda x: is_valid(x), rhythmics))
    if len(rhythmics) == 0:
        continue
    root = roots[rhythmics[0]]
    for i in range(1, len(rhythmics)):
        if root != roots[rhythmics[i]]:
            print('????????')
            print(rhythmics)
    src = ci['paragraphs'][0]
    trg = ''.join(ci['paragraphs'][1:])
    test_data.append({'src':src, 'trg':trg, 'rhythmic':root})

with open('ci.test.json', 'w', encoding="utf-8")  as fout:
    json.dump(test_data, fout, indent=2, ensure_ascii=False)
    