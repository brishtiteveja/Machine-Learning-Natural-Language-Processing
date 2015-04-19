#!/usr/bin/python
import utils
import maxentpostagger
import time
import os, sys
import nltk
import string

from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag import ContextTagger,AffixTagger,RegexpTagger, ClassifierBasedTagger,ClassifierBasedPOSTagger
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.corpus import brown
from utils import str_to_class
from nltk.tag.util import untag
from nltk.metrics.scores import accuracy,precision, recall, log_likelihood, f_measure
from nltk.sem.evaluate import Undefined
  
from nltk.classify.api import ClassifierI, MultiClassifierI
from nltk.classify.weka import WekaClassifier, config_weka
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.positivenaivebayes import PositiveNaiveBayesClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify.util import accuracy, apply_features, log_likelihood
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.maxent import (MaxentClassifier, ConditionalExponentialClassifier)

trainfile="/Users/ananda/Documents/projects/oct27.splits/oct27.traindev"
testfile="/Users/ananda/Documents/projects/oct27.splits/oct27.test"
trainf = open(trainfile, "r")
testf = open(testfile, "r")

#corpus = nltk.corpus.brown.tagged_sents(categories='adventure')[:500]

sents =[]
sent = []
for l in trainf:
    l = l.rstrip()
    if l != "":
        wt = l.split("\t")
        wt[0]=filter(lambda x: x in string.printable, wt[0])
        wt[1]=filter(lambda x: x in string.printable, wt[1])
        w = (wt[0])
        t = (wt[1])
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
        wt[0]=filter(lambda x: x in string.printable, wt[0])
        wt[1]=filter(lambda x: x in string.printable, wt[1])
        w = (wt[0])
        t = (wt[1])
        sent.append((w,t))
    if l == "":
        sents2.append(sent)
#         print len(sents)
#         for item in sents:
#             print item[0], ', '.join(map(str, item[1:]))
        sent=[] 

test = sents2

print len(test)
print ""
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

# print "Unigram Tagger:" 
# print "Training"
# t_s = time.time()
# unigram_tagger = UnigramTagger(training)
#    
# print "Evaluation"
# tagged_sents=unigram_tagger.tag_sents(untag(sent) for sent in test)
# (a,p,r,f) = evaluate(tagged_sents,test)
# print "accuracy: %.1f %%" % (a * 100)
# print "precision: %.1f %%" % (p * 100)
# print "recaull: %.1f %%" % (r * 100)
# print "f_measure: %.1f %%" % (f  * 100)
#    
# t_e = time.time()
# t = t_e - t_s 
# print "Took " + str(t)  + " time"
#    
#    
# print "Bigram Tagger:" 
# print "Training"
# t_s = time.time()
# bigram_tagger = BigramTagger(training, backoff=unigram_tagger) # uses unigram tagger in case it can't tag a word
#    
# print "Evaluation"
# tagged_sents=bigram_tagger.tag_sents(untag(sent) for sent in test)
# (a,p,r,f) = evaluate(tagged_sents,test)
# print "accuracy: %.1f %%" % (a * 100)
# print "precision: %.1f %%" % (p * 100)
# print "recaull: %.1f %%" % (r * 100)
# print "f_measure: %.1f %%" % (f  * 100)
#    
# t_e = time.time()
# t = t_e - t_s 
# print "Took " + str(t)  + " time"
#    
#    
# print "Trigram Tagger:"
# print "Training"
# t_s = time.time()
# trigram_tagger = TrigramTagger(training, backoff=unigram_tagger)
#    
#    
# print "Evaluation"
# tagged_sents=trigram_tagger.tag_sents(untag(sent) for sent in test)
# (a,p,r,f) = evaluate(tagged_sents,test)
# print "accuracy: %.1f %%" % (a * 100)
# print "precision: %.1f %%" % (p * 100)
# print "recaull: %.1f %%" % (r * 100)
# print "f_measure: %.1f %%" % (f  * 100)
# t_e = time.time()
# t = t_e - t_s 
# print "Took " + str(t)  + " time"
#  
# print "Regexp Tagger:"
# print "Training"
# t_s = time.time()
# import string
# regexp_tagger = RegexpTagger(
#             [(r'^-?[0-9]+(.[0-9]+)?$', '$'),   #numberal
#               (r'(The|the|A|a|An|an)$', 'D'),   # articles
#               (r'(^The$|^the$|^A$|^a$|^An$|^an$)$', 'D'),   # articles
#               (r'(Am|am|Is|is|Are|are|Was|was|Were|were|will|shall)$', 'V'),   # auxiliary verbs
#               (r'(^Am$|^am$|^Is$|^is$|^Are$|^are$|^Was$|^was$|^Were$|^were$|^will$|^shall$)$', 'V'),   # auxiliary verbs
#               (r'(^I$|^i$|^me$|^We$|^we$|^us$|^You$|^you$|^He$|^he$|^him$|^her$|^his$|^She$|^she$|^They$|^they$|^Them$|^them$)$', 'D'),   # articles
#               (r'(and|that|but|or|as|if|when|than|because|while|where|after|so|though|since|until|whether|before|although|Although|nor|like|once|unless|until|till|now|except)', '&'),
#               (r'(^and$|^that$|^but$|^or$|^as$|^if$|^when$|^than$|^because$|^while$|^where$|^after$|^so$|^though$|^since$|^until$|^whether$|^before$|^although$|^Although$|^nor$|^like$|^once$|^unless$|^until$|^till$|^now$|^except$)', '&'),
#               (r'(:\-\)|:\-\(|:\-\|\||;\-\)|:\-D:\-\/|:\-P)', 'E'),   # common emoticons
#               (r'^(:\-\)$|^:\-\($|^:\-\|\|$|^;\-\)$|^:\-D$|^:\-\/$|^:\-P$)', 'E'),   # common emoticons
#               (r'^@*$|\s+@*$|@', '@$'),   # @mark
#               (r'^#*$|\s+#*$|^#*$|#', '#'),   # @mark
#               (r'.*\'m$','L'),
#               (r'.*\'s$','L'),
#               (r'.*\'re$','L'),  
#               (r'.^There$', 'X'),  
#               (r'.*able$', 'A'),                # adjectives
#               (r'.*ness$', 'N'),                # nouns formed from adjectives
#               (r'.*ly$', 'R'),                  # adverbs
#               (r'.*ily$', 'R'),                  # adverbs
#               (r'.*ly$', 'R'),                  # adverbs
#               (r'.*ically$', 'R'),                  # adverbs
#               (r'.*lly$', 'R'),                  # adverbs
#               (r'.*es$', 'V'),                  # plural nouns
#               (r'.*ing$', 'V'),
#               (r'.*ed$', 'V'),
#               (r'.*ate$', 'V'),
#               (r'.*en$', 'V'),
#               (r'.*ize$', 'V'),
#               (r'.*ise$', 'V'),
#               (r'.*yze$', 'V'),
#               (r'.*ize$', 'V'),
#               (r'.*ed$', 'V'),
#               (r'.*s$', 'N'),                  # plural nouns
#               (r'.*ing$', 'V'),                # gerunds
#               (r'.*ed$', 'V'),                 # past tense verbs
#               (r'.*ould$', 'V'),
#               (r'.*ness$', 'N'),
#               (r'.*ment$', 'N'),
#               (r'.*al$', 'N'),
#               (r'.*ance$', 'N'),
#               (r'.*ence$', 'N'),
#               (r'.*ation$', 'N'),
#               (r'.*sion$', 'N'),
#               (r'.*tion$', 'N'),
#               (r'.*ure$', 'N'),
#               (r'.*ity$', 'N'),
#               (r'.*age$', 'N'),
#               (r'.*ship$', 'N'),
#               (r'.*acy$', 'N'),
#               (r'.*ability$', 'N'),
#               (r'.*ing$', 'N'),
#               (r'.*ery$', 'N'),
#               (r'.*ful$', 'R'),
#               (r'.*ious$', 'R'),
#               (r'.*ble$', 'A'),
#               (r'.*ic$', 'A'),
#               (r'.*cy$', 'A'),
#               (r'.*ive$', 'A'),
#               (r'.*ic$', 'A'),
#               (r'.*ical$', 'A'),
#               (r'.*ish$', 'A'),
#               (r'.*less$', 'A'),
#               (r'.*like$', 'A'),
#               (r'.*y$', 'A'),
#               (r'.*ous$', 'A'),
#               (r'.*est$', 'A'),
#               (r'^a$', 'P'),
#               (r'^i$', 'P'),
#               (r'['+string.punctuation+']', ','),
#               (r'.*','G')
#          ])
#  
# print "Evaluation"
# tagged_sents=regexp_tagger.tag_sents(untag(sent) for sent in test)
# (a,p,r,f) = evaluate(tagged_sents,test)
# print "accuracy: %.1f %%" % (a * 100)
# print "precision: %.1f %%" % (p * 100)
# print "recaull: %.1f %%" % (r * 100)
# print "f_measure: %.1f %%" % (f  * 100)
# t_e = time.time()
# t = t_e - t_s 
# print "Took " + str(t)  + " time"
# print ""
# 
# print "Affix Tagger:"
# print "Training"
# t_s = time.time()
# affix_tagger = AffixTagger(training,backoff=regexp_tagger)
#      
# print "Evaluation"
# tagged_sents = affix_tagger.tag_sents(untag(sent) for sent in test)
# (a,p,r,f) = evaluate(tagged_sents,test)
# print "accuracy: %.1f %%" % (a * 100)
# print "precision: %.1f %%" % (p * 100)
# print "recaull: %.1f %%" % (r * 100)
# print "f_measure: %.1f %%" % (f  * 100)
# t_e = time.time()
# t = t_e - t_s 
# print "Took " + str(t)  + " time"
#   
# 
# 
# 
# #Various backoff tagger
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
# def train_and_test(est, message):
#     print message
#     hmm = trainer.train_supervised(train_corpus, estimator=est)
#     evl = hmm.evaluate(test_corpus)
#     print evl
#     print evl * 100
#  
# def test_with_different_estimators(ngram=2):
#     #testing
#     mle = lambda fd, bins: MLEProbDist(fd)
#     train_and_test(mle, "MLE")
#     #Laplace (=Lidstone with gamma == 1)
#     train_and_test(LaplaceProbDist, "LaplaceProbDist")
#     #Expected Likelihood Estimation(= Lidstone with gamma == 0.5)
#     train_and_test(ELEProbDist, "ELEProbDist")
#     train_and_test(WittenBellProbDist,"WittenBellProbDist")
#     gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)
#     train_and_test(gt,"SimpleGoodTuringProbDist" )
#     if ngram==3:
#         kn = lambda fd, bins: KneserNeyProbDist(fd)
#         train_and_test(kn, "KneserNeyProbDist")

#stanford log-linear tagger
# from nltk.tag.stanford import POSTagger
# path_to_stanford="/Volumes/Andy's Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Softwares/stanford-postagger/"
# modelfile = path_to_stanford + 'models/twitterModel.tagger'
# jarfile=path_to_stanford +'/stanford-postagger.jar'
#  
#  
#  
# st = POSTagger(modelfile,jarfile) 
# print "Stanford Log Linear Tagger:"
# print "Training"
# t_s = time.time()
# tags = st.tag_sents("I")
# print tags
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

#  Tagging with HMM
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

#HMM tagger with different classifier
# from nltk.probability import (FreqDist, 
#                      ConditionalFreqDist, ConditionalProbDist, 
#                      CrossValidationProbDist,
#                      DictionaryProbDist, DictionaryConditionalProbDist,
#                      ELEProbDist, HeldoutProbDist,
#                      KneserNeyProbDist,
#                      LaplaceProbDist, LidstoneProbDist, 
#                      MutableProbDist, MLEProbDist, 
#                      RandomProbDist, SimpleGoodTuringProbDist,
#                      WittenBellProbDist)
# 
# def train_and_test(est,msg):
#     t_s = time.time()
#     print "Training"
#     hmm_tagger = HiddenMarkovModelTagger.train(training_hmm, estimator=est)
#     print "HMM Tagger(" + msg +"):"
#     print "Evaluation"
#     tagged_sents = hmm_tagger.tag_sents(untag(sent) for sent in test_hmm)
#     (a,p,r,f) = evaluate(tagged_sents,test_hmm)
#     print "accuracy: %.1f %%" % (a * 100)
#     print "precision: %.1f %%" % (p * 100)
#     print "recaull: %.1f %%" % (r * 100)
#     print "f_measure: %.1f %%" % (f  * 100)
#     t_e = time.time()
#     t = t_e - t_s 
#     print "Took " + str(t)  + " time"
#     print " "
#  
# def test_with_different_estimators(ngram=2):
#     #testing
#     #Laplace (=Lidstone with gamma == 1)
#     train_and_test(LaplaceProbDist, "LaplaceProbDist")
#     #Expected Likelihood Estimation(= Lidstone with gamma == 0.5)
#     train_and_test(ELEProbDist, "ELEProbDist")
#     train_and_test(WittenBellProbDist,"WittenBellProbDist")
#     gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)
#     train_and_test(gt,"SimpleGoodTuringProbDist" )
#     if ngram==3:
#         kn = lambda fd, bins: KneserNeyProbDist(fd)
#         train_and_test(kn, "KneserNeyProbDist")
#     mle = lambda fd, bins: MLEProbDist(fd)
#     train_and_test(mle, "MLE")
#     
# 
# training_hmm = training
# test_hmm = test 
#test_with_different_estimators()   

#Trigram HMM
#training_trigram = [[((x[0],y[0],z[0]),(x[1],y[1],z[1]))
#            for x, y, z in nltk.trigrams(sent)]
#                for sent in training]
#training_hmm = training_trigram
#test_hmm = test
#test_with_different_estimators(ngram = 3)

             
                                     
#classifier based tagger
#classifiers = ("NaiveBayesClassifier", "WekaClassifier", "PositiveNaiveBayesClassifier",
#                "DecisionTreeClassifier","MaxentClassifier","SklearnClassifier", "ConditionalExponentialClassifier" )
# classifiers = ("NaiveBayesClassifier", "MaxentClassifier" )
#     
# clsf = [str_to_class(s) for s in classifiers]
#    
#     
# for i in range(len(clsf)):
#      print "Classifier Based Tagger(" + classifiers[i] +"):"
#      print "Training"
#      t_s = time.time()
#      clsfpos_tagger = ClassifierBasedPOSTagger(train=training,classifier_builder=(clsf[i]).train,backoff=regexp_tagger)
#       
#      print "Evaluation"
#      tagged_sents = clsfpos_tagger.tag_sents(untag(sent) for sent in test)
#      (a,p,r,f) = evaluate(tagged_sents,test)
#      print "Classifier Based Tagger(" + classifiers[i] + ") accuracy: %.1f %%" % (a * 100)
#      print "Classifier Based Tagger(" + classifiers[i] + ") precision: %.1f %%" % (p * 100)
#      print "Classifier Based Tagger(" + classifiers[i] + ") recaull: %.1f %%" % (r * 100)
#      print "Classifier Based Tagger(" + classifiers[i] + ") f_measure: %.1f %%" % (f  * 100)
#      t_e = time.time()
#      t = t_e - t_s 
#      print "Took " + str(t)  + " time"
#      print " "


import nltk.tag.crf
os.environ["MALLET"] = "/usr/local/bin/mallet"
os.environ["MALLET_HOME"] = "/usr/local/Cellar/mallet/2.0.7"
os.environ["CLASSPATH"]="$CLASSPATH:/usr/local/Cellar/mallet/2.0.7/libexec/class:/usr/local/Cellar/mallet/2.0.7/libexec/lib/mallet-deps.jar"

nltk.tag.crf.demo()

print "CRF(Conditional Random Field) Tagger:" 
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




#svm_hmm
Tags=['N','O','^','S','Z','V','L','M','A','R','!','D','P','&','T','X','Y','#','#','@','~','U','E','$',',','G']
Features = [
            
            ]

for l in trainf:
    l = l.rstrip()
    if l != "":
        wt = l.split("\t")
        wt[0]=filter(lambda x: x in string.printable, wt[0])
        wt[1]=filter(lambda x: x in string.printable, wt[1])
        w = (wt[0]).decode("utf-8")
        t = (wt[1]).decode("utf-8")
        sent.append((w,t))
    if l == "":
        sents.append(sent)


modelfile = "/Volumes/Andy's Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Softwares/svm_hmm/modelfile.dat"
tagfile = "/Volumes/Andy's Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Softwares/svm_hmm/classify.tags"

path_to_svm_hmm = "/Volumes/Andy's Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Softwares/svm_hmm/"
from subprocess import Popen, PIPE 
print "Training"
t_s = time.time()
process = Popen(["ls", "-l"])
out, err = process.communicate()
process = Popen([path_to_svm_hmm + "svm_hmm_learn","-c 5 -e 0.05 "+ trainfile +" " + modelfile])
out, err = process.communicate()
process = Popen([path_to_svm_hmm + "svm_hmm_classify", testfile +" " + modelfile + tagfile])

tag_file = open(tagfile, "r")
for l in tagfile:
    print l

t_e = time.time()
t = t_e - t_s 
print "Took " + str(t)  + " time"
print " "





