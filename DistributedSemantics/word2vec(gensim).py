import sys,os
import nltk
from nltk.corpus import brown,movie_reviews
import gensim,logging
from gensim import corpora, models, similarities
from gensim import models
from gensim.models import Word2Vec,Doc2Vec

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Sentence Class
class MySentences(object):
     def __init__(self, dirname):
         self.dirname = dirname
 
     def __iter__(self):
         for fname in os.listdir(self.dirname):
             for line in open(os.path.join(self.dirname, fname)):
                 yield line.split()

#sentences = MySentences('/some/directory') # a memory-friendly iterator
#model = gensim.models.Word2Vec(sentences)

BrownModelFileName = "/Users/ananda/Documents/projects/ML-nlp/DistributedSemantics/model/w2v_brown.model"
MovieReviewsModelFileName = "/Users/ananda/Documents/projects/ML-nlp/DistributedSemantics/model/w2v_movie_review.model"
# TreebankModelFileName = "/Users/ananda/Documents/projects/ML-nlp/DistributedSemantics/model/w2v_treebank.model"
# 
# print "Model training started."
# 
# #Train a model from Brown corpus
# sents = brown.sents()
# model_brown = Word2Vec(sents,workers=10)
# 
# print "Saving Brown model."
# model_brown.save(BrownModelFileName)
# 
# #Train a model from treebank corpus
# sents = treebank_raw.sents()
# model_treebank = Word2Vec(sents)
# #Saving model file
# print "Saving Treebank model."
# model_treebank.save(MovieReviewsModelFileName)
# 
# #Train a model from movie review corpus
# sents = movie_reviews.sents()
# model_movie_reviews = Word2Vec(sents)
# 
# print "Saving Movie Reviews model."
# model_movie_reviews.save(MovieReviewsModelFileName)
# 
# 
# print "Model training finished."

#############################################


#Loading model file
print "Loading Model..."
model_brown = Word2Vec.load(BrownModelFileName)
model_movie_reviews = Word2Vec.load(MovieReviewsModelFileName)
print "Models loaded."
print ""

#Test the model
print "Test:Most Similar"
res = model_brown.most_similar(positive=['woman', 'king'], negative=['man'])
print "Brown: pos(woman,king) neg(man)"
print res
print "Brown: great"
res = model_brown.most_similar('great', topn=5)
print res

res = model_movie_reviews.most_similar(positive=['woman', 'king'], negative=['man'])
print "Movie Reviews: pos(woman,king) neg(man)"
print res
print "Movie Reviews: great"
res = model_movie_reviews.most_similar('great', topn=5)
print res

print ""

print "Test:Does not match."
res = model_brown.doesnt_match("breakfast cereal dinner lunch".split())
print "Brown"
print res
res = model_movie_reviews.doesnt_match("breakfast cereal dinner lunch".split())
print "Movie Reviews"
print res

print ""

print "Test:Similarity"
res = model_brown.similarity('woman', 'man')
print "Brown"
print res
res = model_movie_reviews.similarity('woman', 'man')
print "Movie Reviews"
print res

print ""
print "Question-words accuracy"
qfile = "/Users/ananda/Documents/projects/ML-nlp/DistributedSemantics/questions-words.txt" 
print "Brown:"
res = model_brown.accuracy(qfile)
print res
print "Question-words accuracy"
print "Movie Reviews:"
res = model_movie_reviews.accuracy(qfile)
print res
