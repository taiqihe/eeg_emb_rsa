
import sys
import pickle
import os
import re

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
import tensorflow_hub as hub
import numpy as np
from absl import app
from absl import flags

import tokenizer

tf.compat.v1.disable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string('dir', '', 'Folder where the transcripts are stored, and where the embeddings will be saved.', short_name = 'd')

def normalize_sent(line):
	sent = line.replace('-', ' ')
	sent = tokenizer.word_tokenize(sent)
	# Force separation of symbols and letters
	fsent = []
	for w in sent:
		if w.isalpha() or w in {"'s", "'ll", "n't"}:
			fsent.append(w)
		else:
			fsent += re.split(r'(\W+)', w)
	sent = [w.lower() for w in sent if w != '']
	return sent

def main(_):
	elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)
	tokens = tf.placeholder(dtype = tf.string, shape = [None, None])
	tokens_len = tf.placeholder(dtype = tf.int32, shape = [None])
	embeddings = elmo(inputs = {'tokens': tokens, 'sequence_len': tokens_len}, signature = 'tokens', as_dict = True)['elmo']
	init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

	sents = []
	lens = []
	with open(os.path.join(FLAGS.dir, 'TTS_transcript.txt')) as fin:
		for line in fin:
			tl = normalize_sent(line)
			sents.append(tl)
			lens.append(len(tl))

	with open(os.path.join(FLAGS.dir, 'TEC_transcript.txt')) as fin:
		for line in fin:
			tl = normalize_sent(line)
			sents.append(tl)
			lens.append(len(tl))

	maxlen = max([len(sent) for sent in sents])
	for i in range(len(sents)):
		sents[i] += [''] * (maxlen - lens[i])

	config = ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
		sess.run(init_op)

		spb = 100
		sentembs = None
		first = True
		for i in range(0,len(sents), spb):
			right = min(i+spb, len(sents))
			sentin = sents[i:right]
			lensin = lens[i:right]

			batchout = sess.run(embeddings, feed_dict = {tokens: sentin, tokens_len: lensin})
			if not first:
				sentembs = np.append(sentembs, batchout, axis = 0)
			else:
				sentembs = batchout
				first = False

	avgdict = dict()
	for i in range(len(lens)):
		for j in range(lens[i]):
			if sents[i][j] in avgdict:
				avgdict[sents[i][j]].append(sentembs[i][j])
			else:
				avgdict[sents[i][j]] = [sentembs[i][j]]

	with open(os.path.join(FLAGS.dir, 'elmo_averages.txt'), 'w') as fout:
		for w in avgdict:
			avgdict[w] = np.mean(avgdict[w], axis = 0)
			fout.write(w + ' ')
			fout.write(' '.join([str(n) for n in avgdict[w]]))
			fout.write('\n')

	# with open('elmo_averages.pickle', 'wb') as pout:
	# 	pickle.dump(avgdict, file = pout)


if __name__ == '__main__':
	app.run(main)