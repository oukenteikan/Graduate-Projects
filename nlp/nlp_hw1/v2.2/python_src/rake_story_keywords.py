from __future__ import absolute_import
from __future__ import print_function
import six

import rake
import operator
import io, sys, os


def load_stories(infile):
    articles = []
    with open(infile) as inf:
        for line in inf:
            texts = ' '.join(line.strip().split('\t'))
            articles.append(texts)
    print('loaded %d stories!'%len(articles))
    return articles


if __name__ == '__main__':
    # EXAMPLE ONE - SIMPLE
    stoppath = os.path.dirname(os.path.realpath(__file__)) + "/SmartStoplist.txt"
    print (stoppath)
    
    articles = load_stories(sys.argv[1])
    # 1. initialize RAKE by providing a path to a stopwords file
    rake_object = rake.Rake(stoppath, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    with open(sys.argv[5], 'w') as loutf, open(sys.argv[6], 'w') as souf:
        for text in articles: #[:10]:
            keywords = rake_object.run(text, sep=u'[.?!]')
            # 3. print results
            keysents = [k for k, s in keywords.items()]
            sorted_keysents = sorted(keysents, key=operator.itemgetter(1,2))
            #print(sorted_keysents)
            old_sent_id = 0
            for k, si, wi in sorted_keysents:
                loutf.write(k+':'+str(si)+'\t'+str(wi)+'\t')
                if si != old_sent_id:
                    souf.write('#\t')
                    old_sent_id = si
                souf.write(k + '\t')
            loutf.write('\n')
            souf.write('\n')
            #print("-"*86)
