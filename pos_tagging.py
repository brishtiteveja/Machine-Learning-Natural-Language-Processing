#!/usr/bin/python
import numpy as np
import random

#trigram hidden markov model
file_path="/Volumes/Andy's Backup/zehadyzbdullahkhan/Documents/PurdueUniversity/Courses/ML-NLP/Project/Machine-Learning-Natural-Language-Processing/data"
train_file_name=file_path+"/"+"input_string.txt"
train_str_num=0
train_str=[]

#Taking training input strings 
with open(train_file_name, "r") as ins:
    for line in ins:
        train_str_num+=1
        line=line.strip("\n")
        if line.endswith("."):
            line=line[:-1]
        train_str.append(line)

#Taking training input tags
train_tag_file_name=file_path+"/"+"input_tag.txt"
train_tag_str_num=0
train_tag_str=[]
with open(train_tag_file_name, "r") as ins:
    for line in ins:
        train_tag_str_num+=1
        line=line.strip("\n")
        train_tag_str.append(line)

#print train_str_num
#print train_str
#print train_tag_str_num
#print train_tag_str

#getting tag_list
tag_list=[] 
for tags in train_tag_str:
    t = tags.split(" ")
    for tg in t:
        tag_list.append(tg)

#unique tag list
S=sorted(list(set(tag_list)))
S.reverse()
S.append("*")
S.append("*")
S.reverse()
S.append("STOP")

def getTagID(tag_list, tag):
    for i in range(len(S)):
        if S[i] == tag:
            return i
    return -1 #tag doesn't exist in training data
#****************************************************************************************

#parameter estimation for trigram HMM
#Maximum likelihood estimation

#Tag Counting fro trigram HMM
#Count(t_{i-2}, t_{i-1}, t_{i}}
#Count(t_{i-1}, t_{i}}
#Count(t_{i-1}, t_{i}}
tag_len=len(S)
Cnt_1=[0 for p in range(tag_len)] 
Cnt_2=[[0 for p in range(tag_len)] for q in range(tag_len)] 
Cnt_3=[[[0 for p in range(tag_len)] for q in range(tag_len)] for r in range(tag_len)] 
Cnt=0

for tags in train_tag_str:
    t = tags.split(" ")
    l = len(t)
    Cnt+=l
    #print "Tags:"
    #print t
    #print l
    for p in range(l):
        tg1_id=getTagID(tag_list, t[p])
        #print t[p]
        Cnt_1[tg1_id] += 1
        if p < l-1:
            #print (t[p],t[p+1])
            tg2_id=getTagID(tag_list, t[p+1])
            Cnt_2[tg1_id][tg2_id]+=1
        if p < l-2:
            #print (t[p],t[p+1],t[p+2])
            tg2_id=getTagID(tag_list, t[p+1])
            tg3_id=getTagID(tag_list, t[p+2])
            Cnt_3[tg1_id][tg2_id][tg3_id]+=1

#test count
#print "DT NN"
#print Cnt_2[getTagID(tag_list,"DT")][getTagID(tag_list,"NN")]
#print "NN DT"
#print Cnt_2[getTagID(tag_list,"NN")][getTagID(tag_list,"DT")]
#print "NN DT VBZ"
#print Cnt_3[getTagID(tag_list,"NN")][getTagID(tag_list,"DT")][getTagID(tag_list,"VBZ")]
print Cnt

#Calculating transition and emission probabilities
Q=[[[random.random() for p in range(tag_len)] for q in range(tag_len)] for r in range(tag_len)] 
#E=[[random.random() for p in range(tag_len)] for q in range(n)] 
for i in range(2,tag_len):
    for j in range(2,tag_len):
        for k in range(2,tag_len):
#           print Q[i][j][k] 
            a=1
#****************************************************************************************
#Viterbi trigram HMM

def viterbi(s,Q,E): #test sentence, transition probabilities, emission probabilities
    #adding extra two strings in front of the test sentence
    s.reverse()
    s.append("*")
    s.append("*")
    s.reverse()
    n=len(s)

    tag_len=len(S) - 1 
    DP=[[[0 for p in range(tag_len)] for q in range(tag_len)] for r in range(n)] 
    DP[0][0][0] = DP[0][0][1] = DP[0][1][0] = DP[0][1][1] = 1
    BP=[[[0 for p in range(tag_len)] for q in range(tag_len)] for r in range(n)] 

    #temporary test value for the estimates
    Q=[[[random.random() for p in range(tag_len)] for q in range(tag_len)] for r in range(tag_len)] 
    E=[[random.random() for p in range(tag_len)] for q in range(n)] 

    
    for k in range(2,n):
        for u in range(2,tag_len):
            for v in range(2,tag_len):
                maxn = -1 
                maxt = -1
                for w in range(2,tag_len):
                    val = DP[k-1][w][u] * Q[w][u][v] * E[k][v]
                    if val > maxn:
                        maxn=val
                        maxt = w
                DP[k][u][v] = maxn
                BP[k][u][v] = maxt


    print DP
#    print range(tag_len)
#
#    print len(S)
#    print len(DP) 
#    print len(DP[0])
#    print len(DP[0][1])
    #if (k,u,v)==(0,"*","*"):
    #    DP[k][u][v] = 1
    #else:
    #    return 1
         

#viterbi(["hello","me"],"*","*")
