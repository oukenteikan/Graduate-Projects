#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json

rawSongCiDatabase = []

for i in range(0,22):
    with open("../ci.song.{}.json".format(i*1000),"r",encoding="utf-8") as file:
        rawSongCiDatabase.extend(json.load(file))

with open('ci.all.json', 'w', encoding="utf-8")  as fout:
    json.dump(rawSongCiDatabase, fout, indent=2, ensure_ascii=False)

print('合并至ci.all.json')
