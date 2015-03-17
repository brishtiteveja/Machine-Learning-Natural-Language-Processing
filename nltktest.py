#!/usr/bin/python
import nltk
from nltk.tag.stanford import StanfordTagger,POSTagger

from nltk.corpus import brown
from nltk.util import unique_list
from nltk.probability import (FreqDist, 
                     ConditionalFreqDist, ConditionalProbDist, 
                     CrossValidationProbDist,
                     DictionaryProbDist, DictionaryConditionalProbDist,
                     ELEProbDist, HeldoutProbDist,
                     KneserNeyProbDist,
                     LaplaceProbDist, LidstoneProbDist, 
                     MutableProbDist, MLEProbDist, 
                     RandomProbDist, SimpleGoodTuringProbDist,
                     WittenBellProbDist)
from nltk.metrics import accuracy
from nltk.util import LazyMap, unique_list
from nltk.compat import python_2_unicode_compatible, izip, imap 
from nltk.tag.api import TaggerI
 
corpus=[]
tag_set = []
symbols=[]
train_corpus = []
test_corpus = []
 
 
def extractTagSymbol(corpus, tag_set, symbols):
    tag_set += unique_list(tag for sent in corpus for (word,tag) in sent) 
    #print tag_set
    symbols +=unique_list(word for sent in corpus for (word,tag) in sent)
    #print symbols
 
def getTrainTestCorpus(corpus, train_corpus, test_corpus):
    for i in range(len(corpus)):
        if i % 10:
            train_corpus += [corpus[i]]
        else:
            test_corpus += [corpus[i]]
 
def train_and_test(est, message):
    print message
    hmm = trainer.train_supervised(train_corpus, estimator=est)
    evl = hmm.evaluate(test_corpus)
    print evl
    print evl * 100
 
def test_with_different_estimators(ngram=2):
    #testing
    mle = lambda fd, bins: MLEProbDist(fd)
    train_and_test(mle, "MLE")
    #Laplace (=Lidstone with gamma == 1)
    train_and_test(LaplaceProbDist, "LaplaceProbDist")
    #Expected Likelihood Estimation(= Lidstone with gamma == 0.5)
    train_and_test(ELEProbDist, "ELEProbDist")
    train_and_test(WittenBellProbDist,"WittenBellProbDist")
    gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)
    train_and_test(gt,"SimpleGoodTuringProbDist" )
    if ngram==3:
        kn = lambda fd, bins: KneserNeyProbDist(fd)
        train_and_test(kn, "KneserNeyProbDist")
 
#Corpus
corpus = nltk.corpus.brown.tagged_sents(categories='adventure')[:500]
#Extracting tags and words
extractTagSymbol(corpus, tag_set, symbols)
 
 
getTrainTestCorpus(corpus, train_corpus, test_corpus)
#training
trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)
#print "Bigram HMM"
#test_with_different_estimators()
 
train_corpus = []
test_corpus = []
tag_set = []
symbols = []
print len(corpus)
corpus = [[((x[0],y[0],z[0]),(x[1],y[1],z[1]))
            for x, y, z in nltk.trigrams(sent)]
                for sent in corpus[:100]]
extractTagSymbol(corpus, tag_set, symbols)
getTrainTestCorpus(corpus, train_corpus, test_corpus)
#training
trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)
print "Trigram HMM"
test_with_different_estimators(ngram=3)

#stanford log-linear tagger
from nltk.tag.stanford import POSTagger
path_to_stanford="/Volumes/Andy's Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Softwares/stanford-postagger/"
modelfile = path_to_stanford + 'models/english-bidirectional-distsim.tagger'
jarfile=path_to_stanford +'/stanford-postagger.jar'
st = POSTagger(modelfile,jarfile) 
print train_corpus
print test_corpus
Tags = st.tag('What is the airspeed of an unladen swallow ?'.split())
print Tags

