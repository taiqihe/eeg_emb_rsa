import os
import sys
import re
import pickle
import csv

subs = [['1', '2', '3', '4', '5', '6'], ['7'], ['16'], ['28', '30', '40'], ['15', '24', '25', '27'], ['10', '11', '12', '13', '14', '17', '18', '19', '20', '23', '26', '31', '32', '33', '34', '35', '36', '37', '38']]

# sub 6, 14, 12, 23 have multiple boundaries, need to delete the ones that are not separating two stories
# sub 16, 28, 30, 40 need to add 2 between two 1s before story one

def check_consistency(fd):
	for group in subs:
		if len(group) < 2:
			continue
		eseq = []
		for i in group:
			with open(os.path.join(fd, f'elist_{i}.csv')) as cin:
				cr = csv.reader(cin, delimiter = '\t')
				cr.__next__()
				currseq = [line for line in cr if line[1] not in ['-99', '90', '91', '92', '111']]
				if eseq == []:
					eseq = currseq
				else:
					if len(eseq) != len(currseq):
						print(f'Length mismatch for {i}, {len(currseq)} vs {len(eseq)}')
					for j in range(min(len(eseq), len(currseq))):
						if eseq[j][1]!=currseq[j][1]:
							print(f'Eventcode mismatch for {i} at position {currseq[j][0]} for position {eseq[j][0]}')
							break


def prune_story(story):
	with open(f'{story}_words.pickle', 'rb') as pin:
		data = pickle.load(pin)
	nfd = []
	lex = dict()
	for sent in data:
		ns = []
		for w in sent:
			if w[2] - w[1] >= 100: # Min word length
				ns.append(w)
				if w[0] not in lex:
					lex[w[0]] = len(lex)
		nfd.append(ns)
	return nfd, lex


def add_events(folder, wdir):
	stories = ['TEC', 'TTS']
	sdata = [[], []]
	sdict = [[], []]
	sestart = [1000, 3000]
	for s in range(len(stories)):
		sdata[s], sdict[s] = prune_story(os.path.join(wdir, stories[s]))

	with open(os.path.join(wdir, 'BDF_all_words.txt'), 'w') as bout:
		bout.write('Bin 01\nAny word\n.{')
		ecodes = []
		for s in range(len(stories)):
			ecodes += [str(sestart[s] + i) for i in range(len(sdict[s]))]
		bout.write(';'.join(ecodes))
		bout.write('}\n')

	def _is_word_code(code):
		return len(code) == 3 and code.startswith('2')

	def _get_sentence_starts(elist):
		pres = []
		story = -1
		starts = [[], []]
		expect_new_story = True
		for i in range(len(elist)):
			if story == -1:
				if not _is_word_code(elist[i][1]):
					pres.append(elist[i][1])
					continue
				else:
					if len(pres) < 2:
						if len(starts[0]) == 0:
							story = 0
						elif len(starts[1]) == 0:
							story = 1
						else:
							print('Both story have entries when trying to start reading a new one')
							break		
					elif pres[-2] == '2' and len(starts[0]) == 0:
						story = 0
						pres = []
					elif pres[-2] == '1' and len(starts[1]) == 0:
						story = 1
						pres = []
					else:
						print('Something went wrong')
						break
			if story != -1:
				if elist[i][1] in ['200', '99', '88'] and expect_new_story:
					starts[story].append(elist[i+1][4])
					
					# print(elist[i])
					
					expect_new_story = False
				elif elist[i][1] == '201' and expect_new_story:
					starts[story].append(elist[i][4])
					
					# print(elist[i])
					
					expect_new_story = False
				elif _is_word_code(elist[i][1]) and not expect_new_story:
					pass
				elif _is_word_code(elist[i][1]) and expect_new_story:
					starts[story].append(None)
					expect_new_story = False
				elif elist[i][1] == '-99':
					story = -1
					expect_new_story = True
				else:
					expect_new_story = True
		return starts


	flatten_subs = []
	for g in subs:
		flatten_subs = flatten_subs + g

	eformat = '1\t0\t{}\t{}\t{:.3f}\t0\t0\t00000000 00000000\t1\t[]\n'
	for i in flatten_subs:
		with open(os.path.join(folder, f'elist_{i}.csv')) as cin:
			cr = csv.reader(cin, delimiter = '\t')
			cr.__next__()
			currseq = [line for line in cr]
		starts = _get_sentence_starts(currseq)
		if len(starts[0]) != 588 or len(starts[1]) != 528:
			print(f'Subject {i} has incomplete stories, TEC {len(starts[0])}, TTS {len(starts[1])}')

		with open(os.path.join(folder, f'{i}_eventlist_with_words.txt'), 'w') as fout:
			for s in range(len(stories)):
				for j in range(len(starts[s])):
					if starts[s][j] != None:
						for w in range(len(sdata[s][j])):
							fout.write(eformat.format(sestart[s] + sdict[s][sdata[s][j][w][0]], f'{sdata[s][j][w][0]},{sdata[s][j][w][2] - sdata[s][j][w][1]}', sdata[s][j][w][1]/1000+float(starts[s][j]))) # change event code or label here
							# fout.write(eformat.format(sestart[s] + sdict[s][sdata[s][j][w-1][0]], f'{sdata[s][j][w-1][0]},{sdata[s][j][w][2] - sdata[s][j][w][1]}', sdata[s][j][w][1]/1000+float(starts[s][j])))


		


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print(f'Usage: pyhton3 {sys.argv[0]} eventlist_folder working_directory\n')
		print('Assuming pickled generated alignments are stored under the working_directory,\nand will save generated bin definition file under the same folder.')
		exit(1)
	add_events(sys.argv[1], sys.argv[2])
