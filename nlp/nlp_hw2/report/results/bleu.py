#! /usr/bin/env python
# coding=utf-8

import sys
import json
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

res_files = ['same_80.json', 'same_100.json', 'dev1_80.json', 'dev1_100.json', 'dev2_40.json', 'dev2_50.json']
weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
for res_file in res_files: 
    gts_file = 'ci.test.json'

    with open(res_file, 'r') as f:
        res = json.load(f)
        res = [''.join(r['lines'][1:]) for r in res]

    with open(gts_file, 'r') as f:
        gts = json.load(f)
        gts = [g['trg'] for g in gts]

    assert len(res) == len(gts)

    res = [[t for t in r] for r in res]
    gts = [[[t for t in g]] for g in gts]

    for weight in weights:
        corpus_score = corpus_bleu(gts, res, weights=weight)
        sentence_scores = [sentence_bleu(g, r, weights=weight) for (r, g) in zip(res, gts)]
        sentence_score = sum(sentence_scores)
        print('{} Corpus score         {}: {:.5f}'.format(res_file, weights.index(weight)+1, corpus_score))
        print('{} Total sentence score {}: {:.5f}'.format(res_file, weights.index(weight)+1, sentence_score))
