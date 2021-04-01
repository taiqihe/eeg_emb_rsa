import pickle
import re
import random
import os
import sys

import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("eeg_splits", False, "Whether to output EEG files to different splits.")
flags.DEFINE_integer("splits", 32, "Number of parts to randomly split EEGs.")
flags.DEFINE_bool("eeg_indiv", True, "Whether to output per subject EEG files.")
flags.DEFINE_bool("remove_rejected", True, "Whether to remove rejected trials from averages.")
flags.DEFINE_string("rawdir", "raw", "Directory to read raw EEG files and put serialized ouput files.", short_name = 'd')

subs = ['1', '2', '3', '4', '5', '6', '7', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '23', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35', '36', '37', '38', '40']
eegpref = '{}_raw.txt'
epochpref = 'epochs_{}.csv'
rejectpref = 'rejection_{}.csv'

random.seed()

def extract_subs(rawdir = 'raw'):
	splits = FLAGS.splits

	for s in subs:
		print(f'Sub {s}')
		epochs = dict()
		# read in raw EEG
		with open(os.path.join(rawdir, eegpref.format(s))) as eegf:
			time = eegf.readline().rstrip('\n\t').split('\t')[1:]
			epochs['time'] = [float(i) for i in time]
			epochs['eeg'] = []
			for line in eegf:
				curr = line.rstrip('\n\t').split('\t')
				epochs['eeg'].append([float(x) for x in curr[1:]])
		
		# read in epoch information
		with open(os.path.join(rawdir, epochpref.format(s))) as epf:
			epf.readline()
			epochs['info'] = []
			for line in epf:
				res = line.rstrip('\n').split('\t')
				einfo = res[10][3:-1].split(',')
				if not einfo[0][0].isalpha():
					continue
				# print(res)
				
				epochs['info'].append(einfo)

		# read in rejection info
		with open(os.path.join(rawdir, rejectpref.format(s))) as rejf:
			rejf.readline()
			epochs['rejection'] = rejf.read().rstrip('\n ').split('\n')
			epochs['rejection'] = [int(x) for x in epochs['rejection']]

		assert len(epochs['info']) == len(epochs['rejection'])
		trials = len(epochs['info'])
		
		# determine epoch size for data, assuming all epochs have the same size
		epoch_size = 1
		for i in range(1,len(epochs['time'])):
			if epochs['time'][i] > epochs['time'][i-1]:
				epoch_size += 1
			else:
				break
		print(f'epoch size for sub {s} is {epoch_size}')
		
		# check to make sure dimensions agree
		# print(trials, epoch_size, len(epochs['time']))
		assert trials*epoch_size == len(epochs['time'])

		word_averages = dict()
		n_chans = len(epochs['eeg'])
		if FLAGS.eeg_splits or FLAGS.eeg_indiv:
			EEGS = [[] for x in range(splits)]
		epochs['eeg'] = np.asarray(epochs['eeg'])
		for i in range(trials):
			sys.stdout.write(f'\rProcessing {i+1} of {trials}')
			
			e = dict()
			e['sub'] = s
			e['EEG'] = []

			if FLAGS.remove_rejected:
				if epochs['rejection'][i]:
					continue
			else:
				e['rejected'] = epochs['rejection'][i]

			einfo = epochs['info'][i]
			einfo[1] = int(einfo[1])
			ostart = i*epoch_size
			oend = ostart + epoch_size

			# trimming epoch to only include the first word. (disabled)
			# for j in range(ostart, ostart + epoch_size):
			# 	if epochs['time'][j] > einfo[1]:
			# 		oend = j
			# 		break

			e['EEG'] = np.take(epochs['eeg'], range(ostart, ostart+epoch_size), axis = -1)
			e['label'] = einfo[0]

			if e['label'] not in word_averages:
				word_averages[e['label']] = [np.zeros([n_chans, epoch_size]), 0]
			word_averages[e['label']][0] += e['EEG']
			word_averages[e['label']][1] += 1

			if FLAGS.eeg_splits or FLAGS.eeg_indiv:
				spl = random.randrange(splits)
				EEGS[spl].append(e)
		print(f'\n{len(word_averages)} unique words.')

		for w in word_averages:
			word_averages[w] = word_averages[w][0] / word_averages[w][1]
		with open(os.path.join(rawdir, f'word_averages_{s}.pickle'), 'wb') as pout:
			pickle.dump(word_averages, file = pout)
		del word_averages

		if FLAGS.eeg_indiv:
			# Gather all eegs, put in one file
			with open(os.path.join(rawdir, f'EEG_sub_{s}.pickle'), 'wb') as pout:
				pickle.dump([x for y in EEGS for x in y], file = pout)

		if FLAGS.eeg_splits:
			# add eegs to splits
			for spl in range(splits):
				fn = f'EEG_partition_{spl}.pickle'
				if os.path.isfile(fn):
					with open(os.path.join(rawdir, fn), 'rb') as pin:
						odata = pickle.load(pin)
					data = odata + EEGS[spl]
				else:
					data = EEGS[spl]
				with open(os.path.join(rawdir, fn), 'wb') as pout:
					pickle.dump(data, file = pout)

				temp = EEGS[spl]
				EEGS[spl] = []
				del data, temp
		

def shuffle_parts():
	for spl in range(FLAGS.splits):
		fn = f'EEG_partition_{spl}.pickle'
		with open(os.path.join(FLAGS.rawdir, fn), 'rb') as pin:
			data = pickle.load(pin)
		print(f'{len(data)} words in partition {spl}')
		
		random.shuffle(data)
		
		with open(fn, 'wb') as pout:
			pickle.dump(data, file = pout)

def main(_):
	extract_subs(FLAGS.rawdir)
	if FLAGS.eeg_splits:
		shuffle_parts()


if __name__ == '__main__':
	app.run(main)