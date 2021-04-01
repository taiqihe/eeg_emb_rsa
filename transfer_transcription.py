import os
import sys
import re

from absl import app
from absl import flags

import tokenizer

FLAGS = flags.FLAGS
flags.DEFINE_string('transcript', '', 'Transcript file', short_name = 't')
flags.DEFINE_string('sourcedir', '', 'Directory where the TextGrid files are stored', short_name = 'd')
flags.DEFINE_string('prefix', 'TEC', 'Prefix of the TextGrid files', short_name = 'p')


def main(_):
	sents = []
	with open(FLAGS.transcript, 'r') as fin:
		for line in fin:
			sent = line.replace('-', ' ')
			sent = [w for w in tokenizer.word_tokenize(sent) if w not in ['``', "''", '"']]
			sent = [w if w.isalpha() or w in {"'s", "'ll", "n't"} else re.sub('[^a-zA-Z]', '', w) for w in sent]
			sent = [w for w in sent if w != '']
			sents.append(' '.join(sent))

	ftemp = '{}_Sent_{}.TextGrid'
	for i in range(len(sents)):
		with open(os.path.join(FLAGS.sourcedir, ftemp.format(FLAGS.prefix, i+1)), 'r') as tfile:
			cont = tfile.read()
		with open(os.path.join(FLAGS.sourcedir, ftemp.format(FLAGS.prefix, i+1)), 'w') as tfile:
			tfile.write(re.sub('text *= *"[^"]*"', 'text = "{}"'.format(sents[i]), cont))


if __name__ == '__main__':
	app.run(main)