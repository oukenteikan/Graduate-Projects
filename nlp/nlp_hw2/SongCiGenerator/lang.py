import json
import re
import random
import pickle
import numpy
from utils import Lang

rawSongCiDatabase = []
with open("data/ci.pair.pkl","rb") as file:
    rawSongCiDatabase = pickle.load(file)

SongCiDatasets = []
lang = Lang()
for rawSongCi in rawSongCiDatabase:
    SongCi = {}
    lang.addSentence(rawSongCi["src"])
    lang.addSentence(rawSongCi["trg"])
    SongCi["src"] = lang.sentence2Indice(rawSongCi["src"])
    SongCi["trg"] = lang.sentence2Indice(rawSongCi["trg"])
    SongCiDatasets.append(SongCi)

with open("data/SongCiDatasets.pkl","wb") as file:
    pickle.dump(SongCiDatasets,file)
with open("data/lang.pkl","wb") as file:
    pickle.dump(lang,file)

random.shuffle(SongCiDatasets)

train_size = int(0.8 * len(SongCiDatasets))
val_size = int(0.1 * len(SongCiDatasets))
test_size = len(SongCiDatasets) - train_size - val_size

train_set = SongCiDatasets[:train_size]
val_set = SongCiDatasets[train_size:train_size+val_size]
test_set = SongCiDatasets[train_size+val_size:]

with open("data/train_set.pkl","wb") as file:
    pickle.dump(train_set,file)
with open("data/val_set.pkl","wb") as file:
    pickle.dump(val_set,file)
with open("data/test_set.pkl","wb") as file:
    pickle.dump(test_set,file)
