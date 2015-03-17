#!/Users/jogg/anaconda/bin/python

import os
import numpy as np

from collections import Counter

wordfile=open("/Users/jogg/Desktop/Andy/ML-NLP/Data/cvresult0/words.txt", "r")

wordlist = []
for word in wordfile:
   #print word
   wordlist.append(word)  

size=len(wordlist)
wordfreq=Counter(wordlist)

#print wordfreq

highfreqword=wordfreq.most_common(100)
print highfreqword
