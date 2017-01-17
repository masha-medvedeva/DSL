import os
import sys
import time
import itertools
import prep
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix


#from sklearn.naive_bayes import MultinomialNB
from sklearn.base import TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

def dict_words(Xtrain, Ztrain):
	d = {}
	Xtrain_word = [x.split(' ') for x in Xtrain]
	Ztrain_word = [0] * len(Xtrain_word)
	for num in range(len(Xtrain_word)):
		Ztrain_word[num] = [Ztrain[num]] * len(Xtrain_word[num])
	Xtrain_word_flat =  [item for sublist in Xtrain_word for item in sublist]
	Ztrain_flat =  [item for sublist in Ztrain_word for item in sublist]
	for i in range(len(Xtrain_word_flat)):
		if Xtrain_word_flat[i] not in d:
			d[Xtrain_word_flat[i]] = Ztrain_flat[i]
		else:
			d[Xtrain_word_flat[i]] = 0
	return d

def unique_set_generator(*args): # as many arguments as you want, returns the set of unique words to the first language
	for i, bow in enumerate(args):
		rest = (set(x) for j, x in enumerate(args) if j != i)
		rest = set().union(*rest)
		ubow = bow - rest
		yield ubow


class Disjunction(TransformerMixin):
	def __init__(self, d, args):
	    self.d = d
	    self.args = args

	def fit(self, X, y=None):
	    return self

	def transform(self, X):
		print(self.args)
		Xtrain_word = [x.split(' ') for x in X]
		#print(len(Xtrain_word))
		set_un = [[],[],[]]
		count = 0
		for i in self.args:
			for j in self.d:
				if self.d[j] == i:
					set_un[count].append(j)
			count += 1
		set1, set2, set3 = set_un
		set_unique = []
		for unique_set_of_words in unique_set_generator(set(set1), set(set2), set(set3)):
			set_unique.append(unique_set_of_words)

		features = []

		for sentence in Xtrain_word:
			feature = [0, 0, 0]
			for num in range(3):
				for word in sentence:
					if word in set_unique[num]:
						#print (word)
						feature[num] += 1
			k = feature.index(max(feature))
			for i in range(3):
				if i == k:
					feature[i] = 1
				else:
					feature[i] = 0
			features.append(feature)
		#print(features)
		return features

def identity(x):
	return list(itertools.chain.from_iterable(x))


def classify_groups(Xtrain, Ytrain, Xtest, Ytest):
	pipeline = Pipeline([
	('features', FeatureUnion([
		('wordvec', TfidfVectorizer(ngram_range = (1,4), preprocessor = lambda x: x, analyzer = 'word')),
		('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,6), preprocessor = lambda x: x)),
    ])),
    ('classifier', LinearSVC())
    ])
	print('fitting..')
	pipeline.fit(Xtrain, Ytrain)
	Yguess = pipeline.predict(Xtest)
	print('Accuracy: ', accuracy_score(Ytest, Yguess))
	print ('Classification report:\n', classification_report(Ytest, Yguess, labels=["group1", "group2", "group3", "group4", "group5", "group6"]))
	print ('Confusion matrix:\n', confusion_matrix(Ytest, Yguess, labels=["group1", "group2","group3", "group4", "group5", "group6"]))
	return Yguess

def languages_in_groups(group, Xtrain, Ytrain, Ztrain):
	group_lines = []
	group_labels = []
	for i in range(len(Ytrain)):
		if Ytrain[i] == group:
			group_lines.append(Xtrain[i])
			group_labels.append(Ztrain[i])
	return group_lines, group_labels

def classify_within_groups(group, d, Xtrain, Ytrain, Ztrain, Xtest, Ytest_guess, Ztest, Ytest):
	print("GROUP:", group)
	Xtrain_group, Ztrain_group = languages_in_groups(group, Xtrain, Ytrain, Ztrain)
	Xtest_group, Ztest_group = languages_in_groups(group, Xtest, Ytest_guess, Ztest)
	#Xtest_group_true, Ztest_group_true = languages_in_groups(group, Xtest, Ytest, Ztest)

	if group == 'group1':
		args = ['bs','hr','sr']
	elif group == 'group2':
		args = ['my','id']
	elif group == 'group3':
		args = ['fa-AF','fa-IR']
	elif group == 'group4':
		args = ['fr-CA','fr-FR']
	elif group == 'group5':
		args = ['pt-BR','pt-PT']
	elif group == 'group6':
		args = ['es-ES', 'es-AR', 'es-PE']
	pipeline = Pipeline([
	('features', FeatureUnion([
		('wordvec', TfidfVectorizer(ngram_range = (1,8), preprocessor = lambda x: x, analyzer = 'word')),
        #('lang_labels', Disjunction(d, args)),
        ('charvec', TfidfVectorizer(analyzer = 'char', ngram_range = (1,8))),
    ])),
    ('classifier', LinearSVC())
    ])
	print('fitting..')
	pipeline.fit(Xtrain_group, Ztrain_group)
	Zguess_group = pipeline.predict(Xtest_group)
	overall_true.extend(Ztest_group)
	overall_pred.extend(Zguess_group)
	print('True accuracy for this group: ', accuracy_score(Ztest_group, Zguess_group))
	print('Classification report:\n', classification_report(Ztest_group, Zguess_group))
	print ('Confusion matrix:\n', confusion_matrix(Ztest_group, Zguess_group))



if __name__ == '__main__':
	start = time.time()
	#load the data
	Xtrain, Ytrain, Ztrain = prep.extract_words_and_labels('TRAIN')
	Xtest, Ytest, Ztest = prep.extract_words_and_labels('DEV')
	d = dict_words(Xtrain, Ztrain)
	print('METHOD: stage 1, sentence-level, character ngrams')

	#predicted values of the groups
	Ytest_guess = classify_groups(Xtrain, Ytrain, Xtest, Ytest)

	print('METHOD: stage 2-3, sentence-level, character ngrams, no Disjunction')
	groups = ["group1", "group2", "group3", "group4", "group5", "group6"]
	overall_true = []
	overall_pred = []
	for group in groups:

		classify_within_groups(group, d, Xtrain, Ytrain, Ztrain, Xtest, Ytest_guess, Ztest, Ytest)

	end = time.time()
	duration = end - start
print("\n\nOverall accuracy: ", accuracy_score(overall_true, overall_pred))
print('Overall classification report:\n', classification_report(overall_true, overall_pred, labels=['bs','hr','sr','my','id','fa-AF','fa-IR','fr-CA','fr-FR','pt-BR','pt-PT','es-ES','es-AR','es-PE']))
print('Overall confusion matrix:\n', confusion_matrix(overall_true, overall_pred, labels=['bs','hr','sr','my','id','fa-AF','fa-IR','fr-CA','fr-FR','pt-BR','pt-PT','es-ES','es-AR','es-PE']))
print(duration)