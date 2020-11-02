#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import json
import nltk
from tokenize_it import wtokenizer
from util import tokenize, get_noun

class ROCstory:
    def load_data(self, infile):
        dataSet = []
        with open(infile) as inf:
            inf.readline()
            for line in inf:
                elems = line.strip().split('\t')
                try:
                    assert len(elems) == 7 or len(elems) == 8 or len(elems) == 5
                except:
                    print('wrong format!!', len(elems), elems)
                if len(elems) == 7:
                    dataSet.append(elems[2:])
                elif len(elems) == 8:
                    dataSet.append(elems[1:5] + [elems[4+int(elems[-1])]])
                elif len(elems) == 5:
                    dataSet.append(elems)
        assert len(dataSet[-1]) == 5
        print ('Finished loading data!!!')
        return dataSet

    def get_alternative_endings(self, infile, outfile):
        body, trueEnd, falseEnd = [], [], []
        with open(infile) as inf, open(outfile+'.body', 'w') as boutf, open(outfile+'.true', 'w') as toutf, open(outfile+'.false', 'w') as foutf:
            inf.readline()
            for line in inf:
                elems = line.strip().split('\t')
                try:
                    assert len(elems) == 8 
                except:
                    print('wrong format!!', len(elems), elems)
                body.append('\t'.join(elems[1:5]))
                trueEnd.append(elems[4+int(elems[-1])])
                falseEnd.append(elems[7-int(elems[-1])])
            print ('Finished loading data!!!')
            assert len(body) == len(trueEnd)
            assert len(trueEnd) == len(falseEnd)
            for bd, te, fe in zip(body, trueEnd, falseEnd):
                boutf.write(wtokenizer(bd)+'\n')
                toutf.write(wtokenizer(te)+'\n')
                foutf.write(wtokenizer(fe)+'\n')

    
    def gen_pair(self, dataSet, mode='vanilla', look_back=1):
        pairs = []
        keywords = []
        for line in dataSet:
            if mode.startswith('pad'):
                line.insert(0, '<BOST>')
                line.append('<EOST>')
            # The vanilla version
            if mode.endswith('vanilla'):
                pair = (' '.join(line[:-1]).lower(), line[-1].lower())
                pairs.append(pair)
                kws = [get_noun(tokenize(sent)) for sent in line]
                keywords.append(kws)

            # all possible context + ending
            elif mode.endswith('ending'):
                for i in range(len(line)-1):
                    pair = (' '.join(line[i:-1]).lower(), line[-1].lower())
                    pairs.append(pair)
            # all pair
            elif mode.endswith('all'):
                for j in range(1, len(line)):
                    for i in range(j):
                        pair = (' '.join(line[i:j]).lower(), line[j].lower())
                        pairs.append(pair)
            # n-gram
            elif mode.endswith('lookback'):
                # hack for generating test data
                #for j in range(len(line)-1, len(line)):
                for j in range(1, len(line)):
                    i = max(j-look_back, 0)
                    pair = (' '.join(line[i:j]).lower(), line[j].lower())
                    pairs.append(pair)
            else:
                raise NotImplementedError
        print ('generate %d pairs!' % len(pairs))
        return pairs, keywords


    def generate_data(self, infile, outfile, mode='vanilla', look_back=1):
        dataSet = self.load_data(infile)
        pairs, keywords = self.gen_pair(dataSet, mode, look_back)
        with open(outfile+'.tok', 'w') as outf, open(outfile+'.key', 'w') as kof:
            for pair in pairs:
                tpair = map(wtokenizer, pair)
                outf.write('\t'.join(tpair) + '\n')
            for kws in keywords:
                kof.write(' '.join(kws) + '\n')


class VStory(ROCstory):
    def load_data(self, infile):
        dataSet = []
        with open(infile) as inf:
            fobj = json.load(inf)
            #for i in fobj:
            story = fobj['annotations']
            temp_data = []
            for line in story:
                assert len(line) == 1, len(line)
                item_dict = line[0]
                order = item_dict['worker_arranged_photo_order']
                text = item_dict['text']
                assert int(order) == len(temp_data)
                temp_data.append(text)
                if len(temp_data) == 5:
                    dataSet.append(temp_data)
                    temp_data = []
        return dataSet

if __name__ == '__main__':
    infile, outfile, trmode = sys.argv[1:4]
    lb = None
    if len(sys.argv) > 4:
        lb = int(sys.argv[4])
    # generate training data
    rocstory_proc = ROCstory()
    #rocstory_proc.generate_data(infile, outfile, trmode, lb)
    rocstory_proc.get_alternative_endings(infile, outfile)
    #vstory = VStory()
    #vstory.generate_data(infile, outfile, trmode, lb)
