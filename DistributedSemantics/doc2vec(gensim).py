import sys,os
import nltk
from nltk.corpus import brown,movie_reviews
import gensim,logging
from gensim import corpora, models, similarities,utils
from gensim import models
from gensim.models import Word2Vec,Doc2Vec
from gensim.models.doc2vec import LabeledLineSentence, LabeledBrownCorpus,\
    LabeledSentence

logger = logging.getLogger(__name__)

BrownModelFileName = "/Users/ananda/Documents/projects/ML-nlp/DistributedSemantics/model/d2v_brown.model"
MovieReviewsModelFileName = "/Users/ananda/Documents/projects/ML-nlp/DistributedSemantics/model/d2v_movie_review.model"
 
print "Model training started."
 
class LabeledLineSentence(object):
    def __init__(self, sents):
        self.sents = sents
 
    def __iter__(self):
        for uid, wd in enumerate(self.sents):
            yield LabeledSentence(words=wd, labels=['SENT_%s' % uid])

class LabeledSentenceFromCorpusSents(object):
    """Simple format: one sentence = one line = one LabeledSentence object.

    Words are expected to be already preprocessed and separated by whitespace,
    labels are constructed automatically from the sentence line number."""
    def __init__(self, sents):
        """
        `source` can be sentences from available nltk corpus like brown, movie_reviews etc.

        Example::

            sentences = LineSentenceFromCorpus(brown.sents())
            sentences = LineSentenceFromCorpus(movie_reviews.sents())

        """
        self.sents = sents

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Iterate through the sentences in the corpus
            for item_no, wd in enumerate(self.sents):
                yield LabeledSentence(words=utils.to_unicode(wd), labels=['SENT_%s' % item_no])
        except:
            logger.info("Incorrect corpus or source format.")
      
            
#Train a model from Brown corpus
sents = LabeledLineSentence(brown.sents())
model_brown = Doc2Vec(sents,size=100, window=8, min_count=5, workers=4)
  
print "Saving Brown model."
model_brown.save(BrownModelFileName)
  
#Train a model from movie review corpus
sents = LabeledLineSentence(movie_reviews.sents())
model_movie_reviews = Doc2Vec(sents,size=100, window=8, min_count=5, workers=4)
  
print "Saving Movie Reviews model."
model_movie_reviews.save(MovieReviewsModelFileName)
  
print "Model training finished."

print "Loading Model..."
model_brown = Doc2Vec.load(BrownModelFileName)
model_movie_reviews = Doc2Vec.load(MovieReviewsModelFileName)
print "Models loaded."
print ""

#Most Similar
print model_brown.most_similar("SENT_0")
print model_movie_reviews.most_similar("SENT_0")

#Doesn't match
print model_brown.doesnt_match("SENT_0")
print model_movie_reviews.doesnt_match("SENT_0")

#Similarity
print model_brown.similarity("SENT_0","SENT_33450")
print model_brown.similarity("SENT_0","SENT_33451")

