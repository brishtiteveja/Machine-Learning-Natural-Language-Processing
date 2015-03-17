#!/Users/jogg/anaconda/bin/python

import nltk, re, pprint
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import string

from nltk import word_tokenize
from nltk import cluster
from nltk.cluster import util
from nltk.cluster import api
from nltk.cluster import euclidean_distance
from nltk.cluster import cosine_distance
from nltk.corpus import stopwords
from nltk.stem.porter import *

from collections import Counter


from matplotlib.colors import ListedColormap
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA


def BOW(document,chosen_terms):
       word_counts = []
       for word in chosen_terms:
          word_counts.append(document.count(word))
       print word_counts
       return word_counts

def presence(document,chosen_terms):
       word_counts = []
       for word in chosen_terms:
          if document.count(word) == 0:
             word_counts.append(0)
          else:
             word_counts.append(1)
       print word_counts
       return word_counts
    
def TFIDF(document,collectn):
       word_tfidf = []
       for word in chosen_terms:
          word_tfidf.append(collectn.tf_idf(word,document))
       return word_tfidf


DIRLIST=["/Users/jogg/Desktop/Andy/ML-NLP/Data/mix20_rand700_tokens_cleaned/tokens/pos/","/Users/jogg/Desktop/Andy/ML-NLP/Data/mix20_rand700_tokens_cleaned/tokens/neg/"]
cv_cycle=3

for i in range(cv_cycle):
   # print "hello"
   #For positive movie reviews
    DIR=DIRLIST[0]
    os.chdir(DIR)
    filenum=0
    files=[]
    for file in glob.glob("*.txt"):
        filenum+=1
        files.append(file)
    d= int(filenum / cv_cycle)
    #print d

    train_files=[]
    test_files=[]
    #print i
    if i == 0:
       for j in range(2*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)

       for j in range(2*d,3*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  test_files.append(fn)
    elif i==1:
       for j in range(d,3*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)
       #print "\n"
       for j in range(d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  test_files.append(fn)
    elif i==2:
       for j in range(0,d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)
       for j in range(2*d,3*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)
       #print "\n"
       for j in range(d,2*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  test_files.append(fn)

    #print "\n\n"

    def stem_tokens(tokens, stemmer):
       stemmed = []
       for item in tokens:
          stemmed.append(stemmer.stem(item))
       return stemmed

    stemmer = PorterStemmer()

    def processtokens(raw):
        #getting rid of punctuations
        terms_without_punc = [w for w in raw if w not in string.punctuation] 
        #highfreq_terms1=Counter(terms_without_punc).most_common(most_common_param) 
        #print highfreq_terms1
        #print "\n"

        #removing stop words
        terms_withoutstop = [w for w in terms_without_punc if w not in stopwords.words('english')]
        #highfreq_terms2=Counter(terms_withoutstop).most_common(most_common_param)
        #print highfreq_terms2
        #print "\n" 

        #stemming
    	terms_stemmed = stem_tokens(terms_withoutstop,stemmer)
        #highfreq_terms3=Counter(terms_without_punc).most_common(most_common_param) 
        #print highfreq_terms3
        #print "\n"
        
        final_terms=terms_stemmed
        
        return terms_stemmed
 
    chosen_terms=[]
    texts=[]
    texts_train=[]
    texts_test=[]
    raw=""
    #print train_files
    #print test_files
    trainandtestfiles=train_files + test_files
    #print trainandtestfiles

    for file in trainandtestfiles:
        #print file
        f=open(file,"r")
        raw=f.read().decode("latin-1")+"\n"
        f.close()
        pretokens=word_tokenize(raw)
        tokens=processtokens(pretokens)
        chosen_terms=chosen_terms+tokens
        
        #stem the whole text
        tokens=stem_tokens(tokens,stemmer)
        text=nltk.Text(tokens)
        texts.append(text)
        if file in train_files:
           texts_train.append(text)
        elif file in test_files:
           texts_test.append(text)

  # For negative movie reviews
    DIR=DIRLIST[1]
    os.chdir(DIR)
    filenum=0
    files=[]
    for file in glob.glob("*.txt"):
        filenum+=1
        files.append(file)
    d= int(filenum / cv_cycle)
    #print d

    train_files=[]
    test_files=[]
    #print i
    if i == 0:
       for j in range(2*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)

       for j in range(2*d,3*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  test_files.append(fn)
    elif i==1:
       for j in range(d,3*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)
       #print "\n"
       for j in range(d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  test_files.append(fn)
    elif i==2:
       for j in range(0,d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)
       for j in range(2*d,3*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  train_files.append(fn)
       #print "\n"
       for j in range(d,2*d):
          m=re.match(r"cv[0-9]+_tok-([0-9]+).txt",files[j])
          num=m.group(1)
          #print j
          fn = "cv"+'{0:03d}'.format(int(j))+"_tok-"+str(num)+".txt"
          #print fn
	  test_files.append(fn)

    #print "\n\n"


    
    trainandtestfiles=train_files + test_files
    #print trainandtestfiles

    for file in trainandtestfiles:
        #print file
        f=open(file,"r")
        raw=f.read().decode("latin-1")+"\n"
        f.close()
        pretokens=word_tokenize(raw)
        #process_tokens
        tokens=processtokens(pretokens)
        chosen_terms=chosen_terms+tokens

        #stem the whole text
        tokens=stem_tokens(tokens,stemmer)
        text=nltk.Text(tokens)
        texts.append(text)
        if file in train_files:
           texts_train.append(text)
        elif file in test_files:
           texts_test.append(text)
 
    #print texts
    #print texts_train
    #print texts_test
    #print "\n"

   
    #print chosen_terms
    most_common_param=1500
    counts = Counter(chosen_terms).most_common(most_common_param)
    chosen_terms=[ite for ite, it in counts]
    
    print chosen_terms
    print "\n\n"
    
    #chosen_terms=sorted(list(set(all_terms)))
    #print chosen_terms
 
    path="/Users/jogg/Desktop/Andy/ML-NLP/Data/cvresult" + str(i)+"/"
    if not os.path.exists(path):
       #print "hello"
       os.makedirs(path)

    outf=open(path+"words.txt","w")
    for word in chosen_terms:
       #print word
       outf.write(word.encode("latin-1")+"\n")
    
    outf.close
    
    
    #print "Prepared ", len(texts), " documents..."
    #print "They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]"
    
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    collection_train = nltk.TextCollection(texts_train)
    collection_test = nltk.TextCollection(texts_test)
    #print "Created a collection of", len(collection), "terms."
    
    
#################BOW##################

    path=path+"betterfeature/bow/"
    if not os.path.exists(path):
       #print "hello"
       os.makedirs(path)

    trainf=open(path+"train_bow.txt","w")
    testf=open(path+"test_bow.txt","w")

    freq_vectors_bow=[]
    freq_vectors_bow_train=[]
    freq_vectors_bow_test=[]

    freq_vectors = [np.array(BOW(f,chosen_terms)).astype(np.int64) for f in texts]
    #print "Vectors for train files created."
    
    #print freq_vectors_bow

    #print texts_train
    freq_vectors_bow_train = [np.array(BOW(f,chosen_terms)).astype(np.int64) for f in texts_train]
    #print "Vectors for train files created."
    
    #print freq_vectors_bow_train
    np.savetxt(trainf,freq_vectors_bow_train,fmt="%d")
 
    #svm-light train file for bow
    trainf_svm=open(path+"train_bow_svm.txt","w")

    freq_vectors_bow_train_a= np.array(freq_vectors_bow_train)
    (row,col)=freq_vectors_bow_train_a.shape 
    for i in range(row):
        if i in range(2*d):
 	    trainf_svm.write("+1 ")
        else:
 	    trainf_svm.write("-1 ")
        for j in range(col):
            val=freq_vectors_bow_train_a[i,j] 
            if val != 0: 
 	        trainf_svm.write(str(j) + ":" + str(val) + " ")
        trainf_svm.write("\n")
        

    #print texts_test
    freq_vectors_bow_test = [np.array(BOW(f,chosen_terms)).astype(np.int64) for f in texts_test]
    #print "Vectors for test files created."
    
    #print freq_vectors_bow_test
    np.savetxt(testf,freq_vectors_bow_test,fmt="%d")

    #svm-light test file for bow
    testf_svm=open(path+"test_bow_svm.txt","w")

    freq_vectors_bow_test_a= np.array(freq_vectors_bow_test)
    (row,col)=freq_vectors_bow_test_a.shape 
    for i in range(row):
        if i in range(d):
 	    testf_svm.write("+1 ")
        else:
 	    testf_svm.write("-1 ")
        for j in range(col):
            val=freq_vectors_bow_test_a[i,j] 
            if val != 0: 
 	        testf_svm.write(str(j) + ":" + str(val) + " ")
        testf_svm.write("\n")

    trainf.close()
    testf.close()
    trainf_svm.close()
    testf_svm.close()

################PRESENCE##################

    path=path+"../../betterfeature/presence/"
    if not os.path.exists(path):
       #print "hello"
       os.makedirs(path)

    trainf=open(path+"train_presence.txt","w")
    testf=open(path+"test_presence.txt","w")

    freq_vectors_pre=[]
    freq_vectors_pre_train=[]
    freq_vectors_pre_test=[]

    freq_vectors_pre = [np.array(presence(f,chosen_terms)).astype(np.int64) for f in texts]
    #print "Vectors for train files created."
    
    #print freq_vectors_pre

    #print texts_train
    freq_vectors_pre_train = [np.array(presence(f,chosen_terms)).astype(np.int64) for f in texts_train]
    #print "Vectors for train files created."
    
    #print freq_vectors_pre_train
    np.savetxt(trainf,freq_vectors_pre_train,fmt="%d")

    #svm-light train file for presence 
    trainf_svm=open(path+"train_pre_svm.txt","w")

    freq_vectors_pre_train_a= np.array(freq_vectors_pre_train)
    (row,col)=freq_vectors_pre_train_a.shape 
    for i in range(row):
        if i in range(2*d):
 	    trainf_svm.write("+1 ")
        else:
 	    trainf_svm.write("-1 ")
        for j in range(col):
            val=freq_vectors_pre_train_a[i,j] 
            if val != 0: 
 	        trainf_svm.write(str(j) + ":" + str(val) + " ")
        trainf_svm.write("\n")

    #print texts_test
    freq_vectors_pre_test = [np.array(presence(f,chosen_terms)).astype(np.int64) for f in texts_test]
    #print "Vectors for test files created."
    
    #print freq_vectors_pre_test
    np.savetxt(testf,freq_vectors_pre_test,fmt="%d")

    #svm-light test file for bow
    testf_svm=open(path+"test_pre_svm.txt","w")

    freq_vectors_pre_test_a= np.array(freq_vectors_pre_test)
    (row,col)=freq_vectors_pre_test_a.shape 
    for i in range(row):
        if i in range(d):
 	    testf_svm.write("+1 ")
        else:
 	    testf_svm.write("-1 ")
        for j in range(col):
            val=freq_vectors_pre_test_a[i,j] 
            if val != 0: 
 	        testf_svm.write(str(j) + ":" + str(val) + " ")
        testf_svm.write("\n")

    trainf.close()
    testf.close()
    trainf_svm.close()
    testf_svm.close()
    

################TF-IDF##################

    path=path+"../../betterfeature/tf-idf/"
    if not os.path.exists(path):
       #print "hello"
       os.makedirs(path)

    trainf=open(path+"train_tf-idf.txt","w")
    testf=open(path+"test_tf-idf.txt","w")

    freq_vectors_tfidf=[]
    freq_vectors_tfidf_train=[]
    freq_vectors_tfidf_test=[]

    freq_vectors_tfidf = [np.array(TFIDF(f,collection)) for f in texts]
    #print "Vectors for train files created."
    
    #print freq_vectors_tfidf

    #print texts_train
    freq_vectors_tfidf_train = [np.array(TFIDF(f,collection_train)) for f in texts_train]
    #print "Vectors for train files created."
    
    #print freq_vectors_tfidf_train
    np.savetxt(trainf,freq_vectors_tfidf_train,fmt="%.6f")

    #svm-light train file for tf-idf 
    trainf_svm=open(path+"train_tfidf_svm.txt","w")

    freq_vectors_tfidf_train_a= np.array(freq_vectors_tfidf_train)
    (row,col)=freq_vectors_tfidf_train_a.shape 
    for i in range(row):
        if i in range(2*d):
 	    trainf_svm.write("+1 ")
        else:
 	    trainf_svm.write("-1 ")
        for j in range(col):
            val=freq_vectors_tfidf_train_a[i,j] 
            if val != 0: 
 	        trainf_svm.write(str(j) + ":" + str(val) + " ")
        trainf_svm.write("\n")

    #print texts_test
    freq_vectors_tfidf_test = [np.array(TFIDF(f,collection_train)) for f in texts_test]
    #print "Vectors for test files created."
    
    #print freq_vectors_tfidf_test
    np.savetxt(testf,freq_vectors_tfidf_test,fmt="%.6f")

    #svm-light test file for tfidf 
    testf_svm=open(path+"test_tfidf_svm.txt","w")

    freq_vectors_tfidf_test_a= np.array(freq_vectors_tfidf_test)
    (row,col)=freq_vectors_tfidf_test_a.shape 
    for i in range(row):
        if i in range(d):
 	    testf_svm.write("+1 ")
        else:
 	    testf_svm.write("-1 ")
        for j in range(col):
            val=freq_vectors_tfidf_test_a[i,j] 
            if val != 0: 
 	        testf_svm.write(str(j) + ":" + str(val) + " ")
        testf_svm.write("\n")

    trainf.close()
    testf.close()
    trainf_svm.close()
    testf_svm.close()

"""
    #classification
    print(__doc__)



    h = .02  # step size in the mesh

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1), linearly_separable]

    figure = plt.figure(figsize=(27, 9))
    i = 1

classifiers = [KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA()]
h=0.2
# iterate over datasets
for i in range(1):
        X_train, y_train= load_svmlight_file("/Users/jogg/Desktop/Andy/ML-NLP/Data/cvresult0/tf-idf/train_tfidf_svm.txt")
        X_test, y_test= load_svmlight_file("/Users/jogg/Desktop/Andy/ML-NLP/Data/cvresult0/tf-idf/train_tfidf_svm.txt")
        #print X_train
        #print y_train
        #print X_test
        #print y_test
       
        from sklearn import naive_bayes

        mnb = naive_bayes.MultinomialNB()
        ft=mnb.fit(X_train, y_train)       
        print ft
        score=mnb.score(X_test, y_test)
        print score
        #x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        #y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
        y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        ax = plt.subplot(1, len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            ax = plt.subplot(1, len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')
            i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()    
"""
