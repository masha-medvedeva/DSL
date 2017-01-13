from nltk import bigrams, trigrams, ngrams
import re, os, random
import glob


#def extract_labels(lang, lang_words):
#
#	labels = []
#	for num in range(len(lang_words)):
#		labels.append(lang)
#
#	return labels


def extract_words_and_labels(mode):
	all_words = []
	all_labels = []
	all_lines = []
	for file_name in glob.glob('*' + mode + '.txt'):
		#lang = os.path.basename(file_name)[:-4]
		#print('Processing', lang)
		lang_words = []
		labels = []
		#lang_char = []
		f = open(file_name, 'r')
		print(file_name)
		for line in f:
			line = re.sub('\n', '', line)
			line = re.sub('\r\n', '', line)
			#line = re.sub('	', ' ', line)
			line = line.split('	')
			lang_words.extend(line[:-1])
			#lang_char.extend([i for i in line])

		#i2 = []
		#lang_char_4 = list(ngrams(lang_char,3))
		#for i in lang_char_4:
		#	i = list(i)
		#	i = ''.join(i)
		#	i2.append(i)
#
		#lang_char = list(ngrams(i2,5))

			lang = line[-1]
			#print(lang)
			

			for num in range(len(line)-1):
				labels.append(lang)
		all_words.extend(lang_words)

		all_labels.extend(labels)
		#print(all_labels)
	print("Available:", len(all_words))
	assert(len(all_labels)==len(all_words))


	for n in range(len(all_labels)):
		line = (all_words[n], all_labels[n])
		all_lines.append(line)
	random.shuffle(all_lines)
	##list(all_lines)
	all_words = []
	all_labels = []
	for i in all_lines:
		#print (i[0], i[1])
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
		#else:
		#	print('ERROR:', l)
	print('EXAMPLE:', group_labels[0])


	return all_words[:5000], group_labels[:5000], all_labels[:5000]

#def extract_words_and_labels(mode):
#	return extract_words(mode)

