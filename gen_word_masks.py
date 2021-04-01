import sys
import os
import pickle
import collections

import nltk

import tokenizer

def story_to_dict(pref):
	with open(f'{pref}_words.pickle', 'rb') as pin:
		data = pickle.load(pin)

	fd = collections.Counter()
	for s in data:
		fd.update([x[0] for x in s])

	with open(f'{pref}_dict.pickle', 'wb') as pout:
		pickle.dump(fd, file = pout)

	return fd

def printwords(top = None, file = None):
	el = freqdict.most_common(top)
	with open(file, 'w') as fout:
		for w,c in el:
			fout.write(f'{w}\n')

def get_concreteness():
	res = dict()
	with open('function_words.txt') as fin:
		con = set(fin.read().rstrip('\n').split('\n'))
	for w in freqdict:
		if w in con:
			res[w] = 1
		else:
			res[w] = 0
	return res

def get_function_words():
	with open('all_words.txt') as fin:
		words = fin.read().rstrip().split('\n')
	words = nltk.pos_tag(words)
	posdict = dict()
	for w,t in words:
		if t in ['CC', 'DT', 'EX', 'IN', 'LS', 'POS', 'PDT', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB']:
			posdict[w] = 1
		else:
			posdict[w] = 0
	return posdict 

def get_frequency(top = None):
	if top is not None:
		top += 1
	el = freqdict.most_common(top)
	frq = dict()
	for w, c in el:
		frq[w] = c
	return frq

def get_length(top = None):
	words = dict()
	for w, c in freqdict.most_common(top+1):
		words[w] = 0
	with open('TEC_words.pickle', 'rb') as pin:
		data = pickle.load(pin)
	for sent in data:
		for w in sent:
			if w[0] in words:
				words[w[0]] += w[2] - w[1]

	with open('TTS_words.pickle', 'rb') as pin:
		data = pickle.load(pin)
	for sent in data:
		for w in sent:
			if w[0] in words:
				words[w[0]] += w[2] - w[1]
	
	for w in words:
		words[w] = words[w] / freqdict[w]

	return words

def get_embs(top = None):
	keep = freqdict.most_common(top+1)
	keep = set([x[0] for x in keep])
	res = dict()
	with open('all_words_embs.txt') as fin:
		for line in fin:
			w, _, e = line.rstrip('\n').partition(' ')
			if w in keep:
				res[w] = e

	return res

def get_embs_mask(mask):
	with open(mask) as fin:
		include = set(fin.read().rstrip('\n').split('\n'))
	res = dict()
	with open('all_words_embs.txt') as fin:
		for line in fin:
			w, _, e = line.rstrip('\n').partition(' ')
			if w in include:
				res[w] = e
	return res

def get_pos_tags():
	prefs = ['TEC', 'TTS']
	wtags = dict()
	for p in prefs:
		with open(f'{p}_transcript.txt') as fin:
			for line in fin:
				sent = [w for w in tokenizer.word_tokenize(line) if w not in ['``', "''", '"']]
				stags = nltk.pos_tag(sent)
				for w, t in stags:
					wl = w.lower()
					if wl in wtags:
						wtags[wl].append(t)
					else:
						wtags[wl] = [t]
	for w in wtags:
		wtags[w] = collections.Counter(wtags[w]).most_common(1)[0][0]
	printdict(wtags, 'all_words_top_tag.txt')


def get_most_freq_cont_func(top = 100):
	global freqdict
	posd = loaddict('all_words_top_tag.txt', breakarray = True)
	conclist = []
	funclist = []
	for w, f in freqdict.most_common():
		if w not in posd:
			continue
		pos = posd[w]
		if pos.startswith('JJ') or pos.startswith('NN') or pos.startswith('RB') or pos.startswith('VB'):
			if len(conclist) < top:
				conclist.append(w)
		else:
			if len(funclist) < top:
				funclist.append(w)
	with open('top100_func_words.txt', 'w') as fout:
		for w in funclist:
			fout.write(w+'\n')
	with open('top100_conc_words.txt', 'w') as fout:
		for w in conclist:
			fout.write(w+'\n')


def loaddict(file, tofloat = False, breakarray = False):
	data = dict()
	with open(file) as fin:
		for line in fin:
			wl = line.strip().split()
			key = wl[0]
			if tofloat:
				data[key] = [float(x) for x in wl[1:]]
			else:
				data[key] = wl[1:]

			if breakarray:
				data[key] = data[key][0]
	return data

def printdict(d, file):
	with open(file, 'w') as fout:
		for w in d:
			fout.write(f'{w} {d[w]}\n')

if __name__ == '__main__':
	global freqdict
	freqdict = collections.Counter()

	# with open('TEC_dict.pickle', 'rb') as pin:
	# 	data = pickle.load(pin)
	# 	freqdict.update(data)

	# with open('TTS_dict.pickle', 'rb') as pin:
	# 	data = pickle.load(pin)
	# 	freqdict.update(data)
	folder = sys.argv[1]
	freqdict.update(story_to_dict(os.path.join(folder, 'TTS')))
	freqdict.update(story_to_dict(os.path.join(folder, 'TEC')))
	# Because EEGLAB/ERPLAB can't parse a token that starts with "'", discard this word
	freqdict["'s"] = 0
	freqdict["'ll"] = 0

	# get_most_freq_cont_func()

	# printdict(get_frequency(100), 'top100_freq.txt')
	# printdict(get_length(200), 'top200_length.txt')
	# printdict(get_embs(200), 'top200_mbs.txt')

	# printdict(get_embs_mask('words_concrete.txt'), 'concrete_embs.txt')
	# printdict(get_embs_mask('words_nonconcrete.txt'), 'nonconcrete_embs.txt')

	printwords(100, os.path.join(folder, 'top100_words.txt'))

	# printdict(get_concreteness(), 'all_words_functional.txt')