
import os
import sys
import time
#import features
import itertools
import prep

from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, homogeneity_completeness_v_measure, v_measure_score, adjusted_rand_score


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


class grouped_results(TransformerMixin):
    '''Parent class for second order real valued features'''
    def __init__(self, grouped_xtrain):
        self.Xtrain = Xtrain
        self.pos_train = pos_train
        self.pos_test = pos_test
    def fit(self, grouped_xtrain, y=None):
        return self

    def transform(self, X):
        grouped_xtrain = X
        return grouped_xtrain

def identity(x):
    return list(itertools.chain.from_iterable(x))

if __name__ == '__main__': 

	Xtrain, Ytrain, Ztrain = prep.extract_words_and_labels('TRAIN')
	Xtest, Ytest, Ztest = prep.extract_words_and_labels('DEV')
	print('METHOD: stage 1, sentence-level, character ngrams')
	#print(set(Ytrain))


	pipeline = Pipeline([
	('features', FeatureUnion([
        #('wordvec', TfidfVectorizer(ngram_range = (1,3), preprocessor = lambda x: x, tokenizer = identity)),
        ('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,3), preprocessor = lambda x: ' '.join(identity(x)))),
        #('posvecpipeline', Pipeline([('posvec', features.PosVec(Xtrain, pos_train, pos_test)), ('tfidf', TfidfVectorizer(ngram_range = (2,4)))])),
        #('FunctionWords', features.FunctionWords(2500)),
    ])),
    ('classifier', LinearSVC())
    ])  
	print('fitting..')
	pipeline.fit(Xtrain, Ytrain)
	Yguess = pipeline.predict(Xtest)
	print('Accuracy: ', accuracy_score(Ytest, Yguess))
	print ('Classification report:\n', classification_report(Ytest, Yguess, labels=["group1", "group2", "group3", "group4", "group5", "group6"]))
	print ('Confusion matrix:\n', confusion_matrix(Ytest, Yguess, labels=["group1", "group2", "group3", "group4", "group5", "group6"]))
	
	groups = ["group1", "group2", "group3", "group4", "group5", "group6"]
	def languages_in_groups(group, Xtrain, Ytrain, Ztrain):
		group_lines = []
		group_labels = []
		for i in range(len(Xtrain)):
			if Ytrain[i] == group:
				group_lines.append(Xtrain[i])
				group_labels.append(Ztrain[i])
		return group_lines, group_labels
	
	
	print('METHOD: stage 2, sentence-level, character ngrams')
	for group in groups:
		print("GROUP:", group)
		Xtrain2, Ytrain2 = languages_in_groups(group, Xtrain, Ytrain, Ztrain)
		Xtest2, Ytest2 = languages_in_groups(group, Xtest, Ytest, Ztest)
		print(len(Xtest2), len(Ytest2))
		Xtest3 = Xtest2[round(len(Xtest2)/2):]
		Ytest3 = Ytest2[round(len(Ytest2)/2):]

		Xtest2 = Xtest2[:round(len(Xtest2)/2)]
		Ytest2 = Ytest2[:round(len(Ytest2)/2)]
		print(len(Xtest2), len(Ytest2))

#
		pipeline = Pipeline([
		('features', FeatureUnion([
    	    ('wordvec', TfidfVectorizer(ngram_range = (1,8), preprocessor = lambda x: x, tokenizer = identity)),
    	    ('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,8), preprocessor = lambda x: ' '.join(identity(x)))),
    	    #('posvecpipeline', Pipeline([('posvec', features.PosVec(Xtrain, pos_train, pos_test)), ('tfidf', TfidfVectorizer(ngram_range = (2,4)))])),
    	    #('FunctionWords', features.FunctionWords(2500)),
    	])),
    	('classifier', LinearSVC())
    	])  
		print('fitting..')
		pipeline.fit(Xtrain2, Ytrain2)
		Yguess2 = pipeline.predict(Xtest2)
		print('Accuracy: ', accuracy_score(Ytest2, Yguess2))
		print ('Confusion matrix:\n', confusion_matrix(Ytest2, Yguess2))


	#for group in groups:
	#	#print("GROUP:", group)
	#	Xtrain2, Ytrain2 = languages_in_groups(group, Xtrain, Ytrain, Ztrain)
	#	Xtest2, Ytest2 = languages_in_groups(group, Xtest, Ytest, Ztest)
	#	#print(Xtrain2[:10], Xtest2[:10])
		#print[]
		Xtrain2 = [x.split(' ') for x in Xtrain2]
		Xtest2 = [x.split(' ') for x in Xtest2]
		for num in range(len(Xtrain2)):
			Ytrain2[num] = [Ytrain2[num]] * len(Xtrain2[num])
#
		for num in range(len(Xtest2)):
			Ytest2[num] = [Ytest2[num]] * len(Xtest2[num])
#
		#print(Ytest2)

		Xtrain_word =  [item for sublist in Xtrain2 for item in sublist]
		Xtest_word =  [item for sublist in Xtest2 for item in sublist]
		Ytrain_word = [item for sublist in Ytrain2 for item in sublist]
		Ytest_word = [item for sublist in Ytest2 for item in sublist]


		pipeline = Pipeline([
		('features', FeatureUnion([
    	    #('wordvec', TfidfVectorizer(ngram_range = (1,8), preprocessor = lambda x: x, tokenizer = identity)),
    	    ('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,8), preprocessor = lambda x: ' '.join(identity(x)))),
    	    #('posvecpipeline', Pipeline([('posvec', features.PosVec(Xtrain, pos_train, pos_test)), ('tfidf', TfidfVectorizer(ngram_range = (2,4)))])),
    	    #('FunctionWords', features.FunctionWords(2500)),
    	])),
    	('classifier', LinearSVC())
    	])  
		print('fitting..')
		print('results word level')
		pipeline.fit(Xtrain_word, Ytrain_word)
		Yguess_word = pipeline.predict(Xtest_word)


		print('Accuracy: ', accuracy_score(Ytest_word, Yguess_word))
		print ('Confusion matrix:\n', confusion_matrix(Ytest_word, Yguess_word))

	###	grouped_guess= []
	###	count = 0
	###	for arr in Xtest2:
	###		i = len(arr)
	###		grouped_guess.append(Yguess_word[count:count+i])
	###		count += i
###
	###	print('TEST LABELS ONLY')
	###	
	###	#grouped sets of word guesses fit with the actual label of the string
###
	###	#Yguess_sent = pipeline.predict(Xtest_word)
###
###
###
	###	#Xtrain3 = [x.split(' ') for x in Xtrain3]
	###	Xtest3 = [x.split(' ') for x in Xtest3]
	###	#for num in range(len(Xtrain3)):
	###	#	Ytrain3[num] = [Ytrain3[num]] * len(Xtrain3[num])
####
	###	for num in range(len(Xtest3)):
	###		Ytest3[num] = [Ytest3[num]] * len(Xtest3[num])
####
	###	#print(Ytest2)
###
	###	#Xtrain3_word =  [item for sublist in Xtrain3 for item in sublist]
	###	Xtest3_word =  [item for sublist in Xtest3 for item in sublist]
	###	#Ytrain3_word = [item for sublist in Ytrain3 for item in sublist]
	###	Ytest3_word = [item for sublist in Ytest3 for item in sublist]
###
	###	#pipeline.fit(Xtrain3_word, Ytrain3_word)
	###	Yguess_sent = pipeline.predict(Xtest3_word)
###
	###	grouped_guess3= []
	###	count = 0
	###	for arr in Xtest3:
	###		i = len(arr)
	###		grouped_guess3.append(Yguess_sent[count:count+i])
	###		count += i
###
	###	grouped_guess_sh = []
	###	for i in grouped_guess:
	###		print(i)
###
###
	###	Ytest_fuf = [x[0] for x in Ytest2]
	###	#print (len(grouped_guess), len(Ytest_fuf), grouped_guess_sh[:5])
	###	pipeline.fit(grouped_guess_sh, Ytest_fuf)
	###	Yguess_sent = pipeline.predict(grouped_guess3)
###
	###	print('Accuracy: ', accuracy_score(Ytest3, Yguess_sent))
	###	print ('Confusion matrix:\n', confusion_matrix(Ytest3, Yguess_sent))
###
###





#		grouped_xtrain = []
#		count = 0
#		for i in l_xtrain:
#			grouped_xtrain.append(Xtrain2[count:count+i].extend(Ytrain2[count:count+i]))
#			count += i
#			#print (len(Xtrain2[count:count+i]), i)
#
#		#print(len(grouped_xtrain))
#
#		grouped_xtest = []
#		count = 0
#		for i in l_xtest:
#			grouped_xtest.append(Xtest2[count:count+i].extend(Yguess2[count:count+i]))
#			count += i
#
#		grouped_ytrain = []
#		count = 0
#		for i in l_xtrain:
#			grouped_ytrain.append(Ytrain2[count:count+i])
#			count += i
#
#		
#
#
#
#		print("GROUP:", group)
#		Xtrain2, Ytrain2 = languages_in_groups(group, Xtrain, Ytrain, Ztrain)
#		Xtest2, Ytest2 = languages_in_groups(group, Xtest, Ytest, Ztest)
#		#Xtrain2 = grouped_xtrain
#
#		pipeline = Pipeline([
#		('features', FeatureUnion([
#    	    ('wordvec', TfidfVectorizer(ngram_range = (1,8), preprocessor = lambda x: x, tokenizer = identity)),
#    	    ('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,8), preprocessor = lambda x: ' '.join(identity(x)))),
#    	    #('vec', TfidfVectorizer(grouped_xtrain)),
#    	    #('vec', TfidfVectorizer(grouped_ytrain))
#    	    #('FunctionWords', features.FunctionWords(2500)),
#    	])),
#    	('classifier', LinearSVC())
#    	])  
#
#		print('fitting..')
#		print('results sentence level')
#		pipeline.fit(Xtrain2, Ytrain2)
#		Yguess2 = pipeline.predict(Xtest2)
#		print('Accuracy: ', accuracy_score(Ytest2, Yguess2))
#		print ('Confusion matrix:\n', confusion_matrix(Ytest2, Yguess2))
	

