import os
import json
from generateSongCi import generateSongCi, lang
from utils import *


if not os.path.isdir("data"):
    os.makedirs("data")

with open("data/ci.test.json","r") as file:
    testingDataset = json.load(file)

testingOutput = []
cnt = 0
for testCase in testingDataset:
    cnt += 1
    print(cnt)
    result = generateSongCi(testCase["src"],testCase["rhythmic"],lang)
    print(result)
    testingOutput.append(result)

with open("data/testingOutput.json","w") as file:
    json.dump(testingOutput, file, indent=2, ensure_ascii=False)
