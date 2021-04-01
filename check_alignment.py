import sys
import pickle

import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('alignments', '', 'Pickled alignments', short_name = 'a')
flags.DEFINE_string('baseline', '', 'Pickled annotated baseline', short_name = 'b')
flags.DEFINE_bool('print_diffs', False, 'Whether to print diffs of all annotated words', short_name = 'd')


def comp_with_baseline(_):
	with open(FLAGS.alignments, 'rb') as pin:
		alignments = pickle.load(pin)

	with open(FLAGS.baseline, 'rb') as pin:
		baseline = pickle.load(pin)

	for curr in baseline:
		curr['sent'] -= 1
		alsent = alignments[curr['sent']]
		bsent = []
		for i in alsent:
			bsent.append([i[0], -1, -1])
		j = 0
		dis = 0
		for i in range(len(alsent)):
			if abs(curr['stamps'][j] - alsent[i][1]) > dis:
				bsent[i-1][1] = curr['stamps'][j]
				j+=1
				if j >= len(curr['stamps']):
					break
			dis = abs(curr['stamps'][j] - alsent[i][1])
			
			if i == len(alsent)-1:
				if j!=len(curr['stamps'])-1:
					print('Something went wrong during matching')
				bsent[i][1] = curr['stamps'][j]

		curr['stamps'] = bsent 

	devs = []
	for curr in baseline:
		for i in range(len(curr['stamps'])):
			if curr['stamps'][i][1] != -1:
				devs.append(alignments[curr['sent']][i][1] - curr['stamps'][i][1])
				if FLAGS.print_diffs:
					print(f"Sentence {curr['sent']+1} position {i} ({alignments[curr['sent']][i][0]}) error {devs[-1]}")

	print(f'Average error: {np.mean(devs)}')
	print(f'Average absolute error: {np.mean(np.absolute(devs))}')
	print(f'Highest absolute error: {max(np.absolute(devs))}')
	print(f'Longest interval: {max([max([x[2] - x[1] for x in s]) for s in alignments])}')
	print(f'Shortest interval: {min([min([x[2] - x[1] for x in s]) for s in alignments])}')


def get_dict_stats():
	with open(sys.argv[1], 'rb') as pin:
		data = pickle.load(pin)
	lex = dict()
	for sent in data:
		for word in sent:
			if word[0] not in lex:
				lex[word[0]] = 0
			lex[word[0]] += 1

	with open(sys.argv[2], 'wb') as pout:
		pickle.dump(lex, file = pout)


if __name__ == '__main__':
	app.run(comp_with_baseline)