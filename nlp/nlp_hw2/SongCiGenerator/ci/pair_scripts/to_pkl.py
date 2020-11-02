#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import pickle


with open("ci.pair.json","r",encoding="utf-8") as file:
    pairs = json.load(file)
with open('ci.pair.pkl', 'wb') as fout:
    pickle.dump(pairs, fout)

with open("ci.pair.fake.json","r",encoding="utf-8") as file:
    pairs = json.load(file)
with open('ci.pair.fake.pkl', 'wb') as fout:
    pickle.dump(pairs, fout)

with open("ci.pair.all.json","r",encoding="utf-8") as file:
    pairs = json.load(file)
with open('ci.pair.all.pkl', 'wb') as fout:
    pickle.dump(pairs, fout)
