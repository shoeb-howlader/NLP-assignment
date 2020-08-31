# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:00:38 2020

@author: shoeb howlader
"""
#required libraries
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
from nltk import ngrams
from nltk.collocations import *
from nltk.collocations import BigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures 
from bs4 import BeautifulSoup
import sys 
import re
import unidecode as unidecode
from string import digits

#opening file 
text_file = open('bnc.txt')
text=text_file.read()

"""----------------------------------PRE-PROCESSING----------------------------------- """
#removing  html tag
text=re.sub('<[^<]+?>', '', text)

# removing extra white space
text=re.sub(' +', ' ', text)

#removing accented characters
text = unidecode.unidecode(text)

# removing special characters
text = re.sub('[!,*)@#<>|+-=~`/%(&$_?.^]', '', text)

#removing numbers

digits_list = str.maketrans('', '', digits)
text = text.translate(digits_list)
#tokenizing word
tokenized_word=word_tokenize(text)

#removing punctuation marks
tokens_without_punctuations= []
for w in tokenized_word:
    if w.isalpha():
        tokens_without_punctuations.append(w.lower())
        
#sorting alphabetically
tokens_without_punctuations=sorted(tokens_without_punctuations)



"""----------------------------------Question A---------------------------------------------"""
#frequency distribution
fdist = FreqDist(tokens_without_punctuations)
c=1
print("100 most frequent words excluding  puntuations: ",file=open("A_output_for_bnc.txt","w"))
print("word  -->   frequency",file=open("A_output_for_bnc.txt","a"))
top_100=fdist.most_common(100)
for i,j in top_100:
    print(f"{c}. {i} --> {j}",file=open("A_output_for_bnc.txt","a"))
    c+=1



"""---------------------------------Question B ------------------------------------------------"""
#Printing the above information for the next ranks (>100)"""
Rank_list= fdist.most_common(len(tokens_without_punctuations))
rank_greater_than_100=Rank_list[100:]
count=0
print("Words with rank >100 (1000 words per line ):",file=open("B_output_for_bnc.txt","w"))
print("Wrod--> frequency:     ",file=open("B_output_for_bnc.txt","a"))
for x,y in rank_greater_than_100:
    print(f' {x}--> {y}',end =" ",file=open("B_output_for_bnc.txt","a"))
    count+=1
    if count==1000:
        print("\n",file=open("B_output_for_bnc.txt","a"))
        count=0



"""------------------------------------Question C--------------------------------------------------"""
#c1,c2.......................c100"""
frequency_list=[]
for samples in fdist:
    frequency_list.append(fdist[samples])
frequency_list=sorted(frequency_list)
frequency_of_frequency=FreqDist(frequency_list)
print('frequency of frequencies:    \n',file=open("C_output_for_bnc.txt","w"))
for k in range(1,101):
    print(f'C{k} --> {frequency_of_frequency[k]}',file=open("C_output_for_bnc.txt","a"))



    
"""------------------------------------Question D--------------------------------------------------"""
#calculating type/token ratio...........................
types=len(set(tokens_without_punctuations))  #unique_words
tokens= len(tokens_without_punctuations)   #number_of_words
print(f'The corpus has {types} types',file=open("D_output_for_bnc.txt","w"))
print(f'The corpus has {tokens} tokens',file=open("D_output_for_bnc.txt","a"))
print(f'the type/token ratio for the corpus is {types/tokens}',file=open("D_output_for_bnc.txt","a"))




"""------------------------------------Question E--------------------------------------------------"""
#removing stopwords 
stopwords_file = open('stopword.txt')
stopword_text=stopwords_file.read()
#tokenizing stop words
stopwords=word_tokenize(stopword_text)
tokens_without_sw = [word for word in tokens_without_punctuations if not word in stopwords]
fdist_without_sw=FreqDist(tokens_without_sw)
top_100_without_sw=fdist_without_sw.most_common(100)
c=1
print("100 most frequent words excluding stop words and puntuations: ",file=open("E_output_for_bnc.txt","w"))
print("word  -->   frequency",file=open("E_output_for_bnc.txt","a"))
for i,j in top_100_without_sw:
    print(f"{c}. {i} --> {j}",file=open("E_output_for_bnc.txt","a"))
    c+=1



"""------------------------------------Question F--------------------------------------------------"""
#bigrams 
#removing punctuation marks
tokens_without_punct= []
for w in tokenized_word:
    if w.isalpha():
        tokens_without_punct.append(w.lower())
        
#removing stopwords
tokens_without_sw_punct = [word for word in tokens_without_punct if not word in stopwords]
bigram_fd = nltk.FreqDist(nltk.bigrams(tokens_without_sw_punct))

bigram_top_100=bigram_fd.most_common(100)
print("Most frequent word pairs: ",file=open("F_output_for_bnc.txt","w"))
print("Word pair --> frequency",file=open("F_output_for_bnc.txt","a"))
c=1
for x,y in bigram_top_100:
    print(f"{c}. {x} --> {y}",file=open("F_output_for_bnc.txt","a"))
    c+=1


"""------------------------------------Question G--------------------------------------------------"""
#Finding colocation
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
biagram_collocation = BigramCollocationFinder.from_words(tokens_without_sw_punct)
biagram_collocation.apply_freq_filter(3) 

#strong collocation using  mutual information
print("strong collocations of two words using mutual information:",file=open("G_output_for_bnc.txt","w"))
mutual_information = biagram_collocation.nbest(BigramAssocMeasures.pmi, 20)
for z in mutual_information:
    print(z,file=open("G_output_for_bnc.txt","a"))
#strong collocation using  chi-square    
print("\n\nstrong collocations of two words using chi-square: ",file=open("G_output_for_bnc.txt","a"))
chi_square= biagram_collocation.nbest(BigramAssocMeasures.chi_sq, 20)
for v in chi_square:
    print(v,file=open("G_output_for_bnc.txt","a"))

#strong collocation using  likelihood ratio
print("\n\n strong collocations of two words using likelihood ratio:",file=open("G_output_for_bnc.txt","a"))
likelihood_ratio= biagram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 20)
for n in likelihood_ratio:
    print(n,file=open("G_output_for_bnc.txt","a"))

#strong collocation using  phi- square measure
print("\n\n strong collocations of two words using phi- square:",file=open("G_output_for_bnc.txt","a"))
likelihood_ratio= biagram_collocation.nbest(BigramAssocMeasures.phi_sq, 20)
for n in likelihood_ratio:
    print(n,file=open("G_output_for_bnc.txt","a"))
