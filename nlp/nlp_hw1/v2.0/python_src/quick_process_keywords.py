#!/usr/bin/python

import sys

with open(sys.argv[1]) as inf, open(sys.argv[2], 'w') as outf:
    for line in inf:
        keywords = line.strip().split()
        keywords = ' '.join(keywords[:8])
        outf.write(keywords+'\n')

