import os
import sys
import re
import pickle
import csv

# Requires https://github.com/kylebgorman/textgrid
import textgrid
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'alignment', ['baseline', 'alignment'], 'Source of input data')
flags.DEFINE_string('input', None, 'Input data', short_name = 'i')
flags.DEFINE_string('output', None, 'Output pickle file', short_name = 'o')

def from_alignment(ipt, opt):
	for r,d,f in os.walk(ipt):
		files = [os.path.join(r,x) for x in f if x.endswith('.TextGrid')]

	article = [[] for i in range(len(files))]
	for f in files:
		seq = re.search('[0-9]+', f)
		seq = int(seq.group()) -1
		t = textgrid.TextGrid()
		t.read(f)
		for i in t.tiers[0]:
			if i.mark:
				article[seq].append([i.mark, int(i.minTime*1000), int(i.maxTime*1000)])

	with open(opt, 'wb') as pout:
		pickle.dump(article, file = pout)

def from_csv(ipt, opt):
	annotated = []
	with open(ipt, newline = '') as csvfile:
		cr = csv.reader(csvfile, delimiter = ',')
		cr.__next__()
		for row in cr:
			curr = dict()
			curr['sent'] = int(row[1])
			curr['stamps'] = [int(row[x]) for x in range(10, len(row), 2) if row[x]!='']
			annotated.append(curr)
	with open(opt, 'wb') as pout:
		pickle.dump(annotated, file = pout)


def main(_):
	if FLAGS.mode == 'baseline':
		from_csv(FLAGS.input, FLAGS.output)
	else:
		from_alignment(FLAGS.input, FLAGS.output)

if __name__ == '__main__':
	app.run(main)
