#!/usr/bin/python

import sys, re

class IRetriver(object):
    def __init__(self, kfile, cfile):
        self.keywords = self.load_keywords(kfile)
        self.invert_idx = self.build_invert_idx(cfile, self.keywords)

    def load_keywords(self, infile):
        keywords = set()
        with open(infile) as inf:
            for line in inf:
                keywords.update(line.strip().split())
        print ('keywords set size:', len(keywords))
        return keywords


    def build_invert_idx(self, corpus_file, keyword_set):
        invert_idx = dict()
        puncs = set(['.', '?', '!'])
        with open(corpus_file) as cf:
            for line in cf:
                line = line.strip().split()
                sent_idxes = [i for i, w in enumerate(line) if w in puncs]
                try :
                    assert len(sent_idxes) == 5
                except:
                    pass
                    #print('wrong sentence numbers:', len(sent_idxes), line)
                for i, idx in enumerate(sent_idxes):
                    if i == 0:
                        pre_idx = 0
                    else:
                        pre_idx = sent_idxes[i-1]+1
                    sent = line[pre_idx:idx+1]
                    word_set = set(sent)
                    intersection = word_set & keyword_set
                    if len(intersection) != 0:
                        for wd in intersection:
                            if wd not in invert_idx:
                                invert_idx[wd] = []
                            invert_idx[wd].append(sent)
        return invert_idx

    def query(self, keyword):
        return self.invert_idx.get(keyword, [])


if __name__ == '__main__':
    myir = IRetriver(sys.argv[1], sys.argv[2])
    valid_kws = myir.load_keywords(sys.argv[1])
    for kw in valid_kws:
        print('find %d sentences for keyword %s' %(len(myir.query(kw)), kw))
        print(myir.query(kw))
