# -*- coding: utf-8 -*-
import sys
import csv

path = "ROCStories_all_merge_tokenize.titlesepkey."

train_path = path + "train"
test_path = path + "test"
dev_path = path + "dev"

spring_path = "ROCStories_spring.csv"
winter_path = "ROCStories_winter.csv"

train = open(train_path).readlines()
test = open(test_path).readlines()
dev = open(dev_path).readlines()

spring = [row for row in csv.reader(open(spring_path))]
spring = spring[1:]
winter = [row for row in csv.reader(open(winter_path))]
winter = winter[1:]

print(len(train), len(test), len(dev), len(spring), len(winter))

process = train + dev + test
original = spring + winter

print(len(process), len(original))

for i in range(len(process)):
    a = process[i].strip().split('<EOT>')[0].replace(' ', '')
    b = original[i][1].lower().strip().replace(' ', '')
    if a != b:
        print(a, b)
    else:
        if i == len(process) -1 or i % 1000 == 0:
            print("Pass!")

new_train = process[:90002]
new_test = process[90002:]

print(len(new_train), len(new_test))

for i in range(3):
    print(new_train[i])
for i in range(3):
    print(new_train[-i-1])
for i in range(3):
    print(new_test[i])
for i in range(3):
    print(new_test[-i-1])

new_train_path = "roc_key.train"
new_test_path = "roc_key.test"
ftrain = open(new_train_path, 'w')
ftest = open(new_test_path, 'w')
ftrain.writelines(new_train)
ftest.writelines(new_test)
