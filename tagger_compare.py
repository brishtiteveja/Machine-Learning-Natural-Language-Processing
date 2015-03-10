#!/usr/bin/python
import utils
import maxentpostagger
import time
import os, sys
import nltk

from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import ContextTagger,AffixTagger,RegexpTagger, ClassifierBasedTagger,ClassifierBasedPOSTagger
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.corpus import brown
from utils import str_to_class
from nltk.tag.util import untag
from nltk.metrics.scores import accuracy,precision, recall, log_likelihood, f_measure

trainf = open("/Users/ananda/Documents/projects/oct27.splits/oct27.traindev", "r")
testf = open("/Users/ananda/Documents/projects/oct27.splits/oct27.test", "r")

sents =[]
sent = []
for l in trainf:
    l = l.rstrip()
    if l != "":
        wt = l.split("\t")
        w = wt[0]
        t = wt[1]
        sent.append((w,t))
    if l == "":
        sents.append(sent)
#         print len(sents)
#         for item in sents:
#             print item[0], ', '.join(map(str, item[1:]))
        sent=[] 

training = sents
print len(training)

sents2 =[]
sent = []
for l in testf:
    l = l.rstrip()
    if l != "":
        wt = l.split("\t")
        w = wt[0]
        t = wt[1]
        sent.append((w,t))
    if l == "":
        sents2.append(sent)
#         print len(sents)
#         for item in sents:
#             print item[0], ', '.join(map(str, item[1:]))
        sent=[] 

test = sents2


#Brown corpus extraction
# sents = brown.tagged_sents()
# words = brown.tagged_words()
# print "Total sentences in Brown Corpus: " + str(len(sents))
# print "Total words in Brown Corpus: " + str(len(words))
# training = []
# test = []
# 
# for i in range(len(sents)):
#     if i % 10:
#         training.append(sents[i])
#     else:
#         test.append(sents[i])
        

def evaluate(tagged_sents, gold):
    gold_tokens = sum(gold, [])
    test_tokens = sum(tagged_sents, [])
    a = nltk.metrics.scores.accuracy(gold_tokens, test_tokens)
    p = nltk.metrics.scores.precision(set(gold_tokens), set(test_tokens))
    r = nltk.metrics.scores.recall(set(gold_tokens), set(test_tokens))
    f = nltk.metrics.scores.f_measure(set(gold_tokens), set(test_tokens))
    return (a,p,r,f)

print "Unigram Tagger:" 
print "Training"
t_s = time.time()
unigram_tagger = UnigramTagger(training)
  
print "Evaluation"
tagged_sents=unigram_tagger.tag_sents(untag(sent) for sent in test)
(a,p,r,f) = evaluate(tagged_sents,test)
print "accuracy: %.1f %%" % (a * 100)
print "precision: %.1f %%" % (p * 100)
print "recaull: %.1f %%" % (r * 100)
print "f_measure: %.1f %%" % (f  * 100)
  
t_e = time.time()
t = t_e - t_s 
print "Took " + str(t)  + " time"
  
  
print "Bigram Tagger:" 
print "Training"
t_s = time.time()
bigram_tagger = BigramTagger(training, backoff=unigram_tagger) # uses unigram tagger in case it can't tag a word
  
print "Evaluation"
tagged_sents=bigram_tagger.tag_sents(untag(sent) for sent in test)
(a,p,r,f) = evaluate(tagged_sents,test)
print "accuracy: %.1f %%" % (a * 100)
print "precision: %.1f %%" % (p * 100)
print "recaull: %.1f %%" % (r * 100)
print "f_measure: %.1f %%" % (f  * 100)
  
t_e = time.time()
t = t_e - t_s 
print "Took " + str(t)  + " time"
  
  
print "Trigram Tagger:"
print "Training"
t_s = time.time()
trigram_tagger = TrigramTagger(training, backoff=unigram_tagger)
  
  
print "Evaluation"
tagged_sents=trigram_tagger.tag_sents(untag(sent) for sent in test)
(a,p,r,f) = evaluate(tagged_sents,test)
print "accuracy: %.1f %%" % (a * 100)
print "precision: %.1f %%" % (p * 100)
print "recaull: %.1f %%" % (r * 100)
print "f_measure: %.1f %%" % (f  * 100)
t_e = time.time()
t = t_e - t_s 
print "Took " + str(t)  + " time"
 
print "Regexp Tagger:"
print "Training"
t_s = time.time()
import string
regexp_tagger = RegexpTagger(
            [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
              (r'(The|the|A|a|An|an)$', 'AT'),   # articles
              (r'.*able$', 'JJ'),                # adjectives
              (r'.*ness$', 'NN'),                # nouns formed from adjectives
              (r'.*ly$', 'RB'),                  # adverbs
              (r'.*s$', 'NNS'),                  # plural nouns
              (r'.*ing$', 'VBG'),                # gerunds
              (r'.*ed$', 'VBD'),                 # past tense verbs
              (r'.*', 'NN'),                      # nouns (default)
              (r'.*ould$', 'MD'),
              (r'.*ing$', 'VBG'),
              (r'.*ed$', 'VBD'),
              (r'.*ness$', 'NN'),
              (r'.*ment$', 'NN'),
              (r'.*ful$', 'JJ'),
              (r'.*ious$', 'JJ'),
              (r'.*ble$', 'JJ'),
              (r'.*ic$', 'JJ'),
              (r'.*ive$', 'JJ'),
              (r'.*ic$', 'JJ'),
              (r'.*est$', 'JJ'),
              (r'^a$', 'PREP'),
              (r'^i$', 'PN'),
              (r'^[A-Z][a-z]$', 'PN'),
              (r'['+string.punctuation+']', 'PUN')
         ])
 
print "Evaluation"
tagged_sents=regexp_tagger.tag_sents(untag(sent) for sent in test)
(a,p,r,f) = evaluate(tagged_sents,test)
print "accuracy: %.1f %%" % (a * 100)
print "precision: %.1f %%" % (p * 100)
print "recaull: %.1f %%" % (r * 100)
print "f_measure: %.1f %%" % (f  * 100)
t_e = time.time()
t = t_e - t_s 
print "Took " + str(t)  + " time"
print ""

print "Affix Tagger:"
print "Training"
t_s = time.time()
affix_tagger = AffixTagger(training,backoff=regexp_tagger)
     
print "Evaluation"
tagged_sents = affix_tagger.tag_sents(untag(sent) for sent in test)
(a,p,r,f) = evaluate(tagged_sents,test)
print "accuracy: %.1f %%" % (a * 100)
print "precision: %.1f %%" % (p * 100)
print "recaull: %.1f %%" % (r * 100)
print "f_measure: %.1f %%" % (f  * 100)
t_e = time.time()
t = t_e - t_s 
# print "Took " + str(t)  + " time"
  
  
from nltk.classify.api import ClassifierI, MultiClassifierI
from nltk.classify.weka import WekaClassifier, config_weka
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.positivenaivebayes import PositiveNaiveBayesClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify.util import accuracy, apply_features, log_likelihood
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.maxent import (MaxentClassifier, ConditionalExponentialClassifier)
                                     
#classifiers = ("NaiveBayesClassifier", "WekaClassifier", "PositiveNaiveBayesClassifier",
#                "DecisionTreeClassifier","MaxentClassifier","SklearnClassifier", "ConditionalExponentialClassifier" )
classifiers = ("DecisionTreeClassifier","MaxentClassifier","SklearnClassifier", "ConditionalExponentialClassifier" )
   
clsf = [str_to_class(s) for s in classifiers]
  
   
for i in range(len(clsf)):
    print "Classifier Based Tagger(" + classifiers[i] +"):"
    print "Training"
    t_s = time.time()
    clsfpos_tagger = ClassifierBasedPOSTagger(train=training,classifier_builder=(clsf[i]).train,backoff=regexp_tagger)
     
    print "Evaluation"
    tagged_sents = clsfpos_tagger.tag_sents(untag(sent) for sent in test)
    (a,p,r,f) = evaluate(tagged_sents,test)
    print "Classifier Based Tagger(" + classifiers[i] + ") accuracy: %.1f %%" % (a * 100)
    print "Classifier Based Tagger(" + classifiers[i] + ") precision: %.1f %%" % (p * 100)
    print "Classifier Based Tagger(" + classifiers[i] + ") recaull: %.1f %%" % (r * 100)
    print "Classifier Based Tagger(" + classifiers[i] + ") f_measure: %.1f %%" % (f  * 100)
    t_e = time.time()
    t = t_e - t_s 
    print "Took " + str(t)  + " time"
    print " "


#Various backoff tagger
# def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
#     if not backoff:
#         backoff = tagger_classes[0](tagged_sents)
#         del tagger_classes[0]
#  
#     for cls in tagger_classes:
#         tagger = cls(tagged_sents, backoff=backoff)
#         backoff = tagger
#  
#     return backoff
# 
# 
# seq_taggers = ["ubt_tagger", "utb_tagger", "but_tagger", "btu_tagger", "tub_tagger", "tbu_tagger",
#                 "ubta_tagger", "ubat_tagger", "uabt_tagger", "aubt_tagger", "raubt_tagger"]
# 
# def ubt_tagger():
#     return backoff_tagger(training, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger])
# def utb_tagger(): 
#     return backoff_tagger(training, [nltk.tag.UnigramTagger, nltk.tag.TrigramTagger, nltk.tag.BigramTagger])
# def but_tagger():
#     return backoff_tagger(training, [nltk.tag.BigramTagger, nltk.tag.UnigramTagger, nltk.tag.TrigramTagger])
# def btu_tagger():
#     return backoff_tagger(training, [nltk.tag.BigramTagger, nltk.tag.TrigramTagger, nltk.tag.UnigramTagger])
# def tub_tagger():
#     return backoff_tagger(training, [nltk.tag.TrigramTagger, nltk.tag.UnigramTagger, nltk.tag.BigramTagger])
# def tbu_tagger():
#     return backoff_tagger(training, [nltk.tag.TrigramTagger, nltk.tag.BigramTagger, nltk.tag.UnigramTagger])
# def ubta_tagger():
#     return backoff_tagger(training, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger, nltk.tag.AffixTagger])
# def ubat_tagger():
#     return backoff_tagger(training, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.AffixTagger, nltk.tag.TrigramTagger])
# def uabt_tagger():
#     return backoff_tagger(training, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.AffixTagger, nltk.tag.TrigramTagger])
# def aubt_tagger():
#     return backoff_tagger(training, [nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.AffixTagger, nltk.tag.TrigramTagger])
# def raubt_tagger():
#     return backoff_tagger(training, [nltk.tag.AffixTagger, nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
#     backoff=regexp_tagger)
# 
# 
# for i in range(len(seq_taggers)):
#     print "Sequence Tagger(" + seq_taggers[i] +"):"
#     print "Training"
#     t_s = time.time()
#     methodToCall = getattr(sys.modules[__name__],seq_taggers[i])
#     seqt_tagger = methodToCall();
#     print "Evaluation"
#     tagged_sents = seqt_tagger.tag_sents(untag(sent) for sent in test)
#     (a,p,r,f) = evaluate(tagged_sents,test)
#     print "SequenceTagger(" + seq_taggers[i] + ") accuracy: %.1f %%" % (a * 100)
#     print "SequenceTagger(" + seq_taggers[i] + ") precision: %.1f %%" % (p * 100)
#     print "SequenceTagger(" + seq_taggers[i] + ") recaull: %.1f %%" % (r * 100)
#     print "SequenceTagger(" + seq_taggers[i] + ") f_measure: %.1f %%" % (f  * 100)
#        
#     t_e = time.time()
#     t = t_e - t_s 
#     print "Took " + str(t)  + " time"
#     print " "
# 
# 
#  
# print "HMM Tagger:"
# print "Training"
# t_s = time.time()
# hmm_tagger = HiddenMarkovModelTagger.train(training)
#  
# print "Evaluation"
# tagged_sents=hmm_tagger.tag_sents(untag(sent) for sent in test)
# (a,p,r,f) = evaluate(tagged_sents,test)
# print "accuracy: %.1f %%" % (a * 100)
# print "precision: %.1f %%" % (p * 100)
# print "recaull: %.1f %%" % (r * 100)
# print "f_measure: %.1f %%" % (f  * 100)
# t_e = time.time()
# t = t_e - t_s 
# print "Took " + str(t)  + " time"