import re, os, random
import glob
#pseudo-random, keep the number constant to be able to compare results
random.seed(132)

def extract_words_and_labels(mode):
	all_words = []
	all_labels = []
	all_lines = []
	for file_name in glob.glob('*' + mode + '.txt'):
		lang_words = []
		labels = []
		f = open(file_name, 'r')
		print(file_name)
		for line in f:
			line = re.sub('\n', '', line)
			line = re.sub('\r\n', '', line)
			line = line.split('	') #split by the tab between the line and the language
			#the line without the language
			lang_words.extend(line[:-1])
			## split the lines by space, strip of everythin, lower, bla bla and reassemble again?
			#language
			lang = line[-1]
			
			for num in range(len(line)-1):
				labels.append(lang)
		all_words.extend(lang_words)
		all_labels.extend(labels)

	assert(len(all_labels)==len(all_words))


	for n in range(len(all_labels)):
		line = (all_words[n], all_labels[n])
		all_lines.append(line)
	#shuffle the lines for training, otherwise it learns 'bs' only first and then 'hr' only	
	random.shuffle(all_lines)
	all_words = []
	all_labels = []
	for i in all_lines:
		all_words.append(i[0])
		all_labels.append(i[1])

	group_labels = []
	for l in all_labels:
		if l == 'bs' or l == 'hr' or l == 'sr':
			group_labels.append('group1')
		if l == 'my' or l == 'id':
			group_labels.append('group2')
		if l == 'fa-AF' or l == 'fa-IR':
			group_labels.append('group3')
		if l == 'fr-CA' or l == 'fr-FR':
			group_labels.append('group4')
		if l == 'pt-BR' or l == 'pt-PT':
			group_labels.append('group5')
		if l == 'es-ES' or l == 'es-AR' or l == 'es-PE':
			group_labels.append('group6')

	#Xtrain, Ytrain, Ztrain
	#sentence, group, language
	return all_words, group_labels, all_labels
	#return all_words[:5000], group_labels[:5000], all_labels[:5000]
	#try out on smaller dataset, 5000 is the length of test data
