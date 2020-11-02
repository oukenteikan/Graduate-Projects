#!/usr/bin/python
import numpy

sent_dict = {0:'*HappyEnding', 1:'*SadEnding'}

spirit_array = numpy.random.binomial(1, .5, size=1871)

for s in spirit_array:
    print(sent_dict[s])
