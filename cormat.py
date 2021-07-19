
# Main script
import sys
import os
from multiprocessing import Pool
import itertools
import operator
import collections
import pickle
import random
import datetime
import copy

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt
from absl import app
from absl import flags

import mystats
import helpers

FLAGS = flags.FLAGS

# For convenience, hard-code subject numbers
subs = ['1', '2', '3', '4', '5', '6', '7', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '23', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35', '36', '37', '38', '40']

# Define key time points
PRE = 100
ONSET = 150
MAXLEN = 400

# For manual use only
# dataloc = '../../ContinuousStory/raw/word_averages_{}.pickle'
# rawdata_fmt = '../../ContinuousStory/raw/EEG_sub_{}.pickle'


# Utilities for computing various correlation matrices

def erp_cor(arg):
	x, y = arg
	global ARR
	return stats.pearsonr(ARR[x], ARR[y])[0]

def emb_cor(arg):
	x, y = arg
	global ARR
	return np.dot(ARR[x],ARR[y])

def meas_cor(arg):
	x, y = arg
	global ARR
	return ARR[x] - ARR[y]

def binary_cor(arg):
	x, y = arg
	global ARR
	return ARR[x] == ARR[y]

def worker_init(arr):
	global ARR
	ARR = arr

def corr_mat(data, pfunc, ids, w2idx = None):
	if w2idx is not None:
		remap = [[] for i in range(len(w2idx))]
		for w in w2idx:
			remap[w2idx[w]] = np.ravel(data[w])
		ARR = remap
	else:
		ARR = data

	with Pool(16, initializer = worker_init, initargs = (ARR,)) as p:
		ans = p.map(pfunc, itertools.combinations(ids, 2))

	return ans


# Step-wise linear regression
def linreg_steps(Xs, Y, combs):

	def _linreg(X, Y):
		reg = LinearRegression()
		reg.fit(X,Y)
		return [reg.score(X,Y), reg.coef_]

	ans = dict()
	for c in combs:
		cx = np.take(Xs, c, axis = 0).transpose()
		ans[c] = _linreg(cx, Y)

	return ans


# Select words with at least [cutoff] instances and randomly sample the rest
# of data to have the same number of instances (minimum frequency of the
# remaining samples)
def equal_freq_sample(EEGs, wordlist, cutoff = 30):
	sample = dict()
	for e in EEGs:
		w = e['label']
		if w in wordlist:
			if w not in sample:
				sample[w] = []
			sample[w].append(e['EEG'])
	
	resample = dict()
	mf = len(EEGs)
	for w in sample:
		if len(sample[w]) >= cutoff:
			resample[w] = sample[w]
			if len(sample[w]) < mf:
				mf = len(sample[w])
	sample = resample
	print(f'Minimum frequency is {mf}')
	
	rng = np.random.default_rng()
	for w in sample:
		eegs = sample[w]
		sample[w] = np.sum(rng.choice(eegs, mf, replace = True), axis = 0) / mf
	return sample


def remove_average_erp(data, baseline_corr = True, partial = None):
	# get data dict from one sub, remove average for each channel
	# partial sets the boundaries of subinterval to use
	if baseline_corr:
		if partial:
			for w in data:
				data[w] = partial_baseline_correction(data[w], partial[0], partial[1])[:29]
		else:
			for w in data:
				data[w] = baseline_correction(data[w], PRE)[:29] # use only first 29 channels

	avg = np.sum(list(data.values()), axis = 0) / len(data)
	for w in data:
		data[w] -= avg

	return data

def baseline_correction_erp(data, partial = None):	
	if partial:
		for w in data:
			data[w] = partial_baseline_correction(data[w], partial[0], partial[1])[:29]
	else:
		for w in data:
			data[w] = baseline_correction(data[w], PRE)[:29] # use only first 29 channels

	return data

def baseline_correction(EEG, zeroloc):
	bsl = np.mean(np.take(EEG, range(zeroloc), axis = -1), axis = -1, keepdims = True)
	ans = EEG-bsl
	return ans

def partial_baseline_correction(EEG, start, end):
	bsl = np.mean(np.take(EEG, range(start, end), axis = -1), axis = -1, keepdims = True)
	ans = EEG-bsl
	return ans

def fdr_correction(h, p, alpha):
	# use the Benjamini-Hochberg procedure to control the FDR at level alpha
	# h is a list of values, p is a list of p values
	if len(h) != len(p):
		print('lenghts of p and h must agree')
		return None
	ph = [[p[i], h[i]] for i in range(len(h))]
	ph = sorted(ph, key = operator.itemgetter(0)) # prevent secondary sorting by h
	for k in range(len(ph)-1, -1, -1):
		if p[k] <= alpha * (k+1) / len(ph):
			break
	return h[:k]


def load_word_embs(file = 'all_words_embs.txt', normalize = True, discard_firstln = False):
	word_embs = dict()
	print(f'Loading word embeddings from {file}')
	with open(file) as ein:
		if discard_firstln:
			ein.readline()
		for line in ein:
			lp = line.rstrip('\n ').split(' ')
			word_embs[lp[0]] = np.asarray([float(x) for x in lp[1:]])
			if normalize:
				word_embs[lp[0]] = word_embs[lp[0]] / np.linalg.norm(word_embs[lp[0]])

	return word_embs


# randomly permute the embeddings
def scramble_embs(embs):
	random.seed()
	ks = list(embs.keys())
	random.shuffle(ks)
	nembs = dict()
	for i, w in enumerate(embs):
		nembs[ks[i]] = embs[w]

	return nembs


def get_erpavg(subl):
	gaerps = dict()
	for s in subl:
		with open(dataloc.format(s), 'rb') as pin:
			erps = pickle.load(pin)

		for w in erps:
			if w not in gaerps:
				gaerps[w] = [np.ravel(erps[w]), 1]
			else:
				gaerps[w][0] += np.ravel(erps[w])
				gaerps[w][1] += 1
	for w in gaerps:
		gaerps[w] = gaerps[w][0] / gaerps[w][1]
	return gaerps


def get_split_erp(data):
	erp = dict()
	for d in data:
		if d['label'] not in erp:
			erp[d['label']] = [d['EEG'].copy()]
		else:
			erp[d['label']].append(d['EEG'].copy())

	erp1 = dict()
	erp2 = dict()
	for w in erp:		
		part1 = random.sample(range(len(erp[w])), int(len(erp[w])/2))
		part2 = [x for x in range(len(erp[w])) if x not in part1]
		if len(part1) > 0:
			part1 = np.asarray(np.sum(np.take(erp[w], part1, axis = 0), axis = 0)) / len(part1)
			erp1[w] = np.squeeze(part1)

		if len(part2) > 0:
			part2 = np.asarray(np.sum(np.take(erp[w], part2, axis = 0), axis = 0)) / len(part2)
			erp2[w] = np.squeeze(part2)

	erp1 = remove_average_erp(erp1)
	erp2 = remove_average_erp(erp2)
	return erp1, erp2


def get_noise_ceilings():
	mask = load_word_embs(FLAGS.mask, False)
	mask = set(mask.keys())

	subrs = dict()
	allrs = None
	for s in subs:
		with open(os.path.join(FLAGS.rawdata, 'word_averages_{}.pickle').format(s), 'rb') as pin:
			erps = pickle.load(pin)
		erps = remove_average_erp(erps, partial = (PRE, ONSET))

		w2idx = dict()
		i = 0
		for w in set(erps.keys()) & mask:
			w2idx[w] = i
			i+=1

		print('Sub {}: Found {} words in common'.format(s, len(w2idx)))

		ids = range(len(w2idx))

		postbsl = {k: np.take(x, range(PRE, MAXLEN), axis = -1) for k, x in erps.items()}
		erpcor = np.array(corr_mat(postbsl, erp_cor, ids, w2idx))
		subrs[s] = erpcor
		if allrs is None:
			allrs = np.copy(erpcor)
		else:
			allrs += erpcor

	logf = open('noise_ceilings_{}.csv'.format(datetime.date.today().strftime('%m%d%y')), 'w')
	for s in subs:
		ub, p = stats.spearmanr(subrs[s], allrs/len(subrs))
		lb, p = stats.spearmanr(subrs[s], (allrs - subrs[s]) / (len(subrs) -1))
		# ub = mystats.tau_a(subrs[s], allrs/len(subrs))
		# lb = mystats.tau_a(subrs[s], (allrs - subrs[s]) / (len(subrs) -1))
		logf.write('{},{},{}\n'.format(s, lb, ub))
	logf.close()


def proc_indiv(log_file = 'rs_top100_elmo_cont_{}.csv', plot_cormat = False, equal_sample = 0, noise_ceil = ''):
	wordembs = load_word_embs(FLAGS.embs, True)
	if FLAGS.mask != '':
		mask = load_word_embs(FLAGS.mask, False)
		mask = set(mask.keys())
	else:
		mask = None

	logf = open(log_file.format(datetime.date.today().strftime('%m%d%y')), 'w')
	
	allrs = None
	subrs = dict()
	for s in subs:
		if equal_sample > 0:
			with open(os.path.join(FLAGS.rawdata, 'EEG_sub_{}.pickle').format(s), 'rb') as pin:
				eegs = pickle.load(pin)
			erps = equal_freq_sample(eegs, mask, equal_sample)
		else:
			with open(os.path.join(FLAGS.rawdata, 'word_averages_{}.pickle').format(s), 'rb') as pin:
				erps = pickle.load(pin)
			erps = baseline_correction_erp(erps, partial = (PRE, ONSET))
		
		w2idx = dict()
		i = 0
		if mask != None:
			union = set(erps.keys()) & set(wordembs.keys()) & mask
		else:
			union = set(erps.keys()) & set(wordembs.keys())
		for w in union:
			w2idx[w] = i
			i+=1

		print('Sub {}: Found {} words in common'.format(s, len(w2idx)))

		ids = [i for i in range(len(w2idx))]

		embcor = corr_mat(wordembs, emb_cor, ids, w2idx)

		postbsl = {k: np.take(x, range(ONSET, MAXLEN), axis = -1) for k, x in erps.items()}
		# helpers.test_sub_stats(postbsl)
		erpcor = corr_mat(postbsl, erp_cor, ids, w2idx)

		if plot_cormat:
			freql = [freq[w] for w in w2idx]
			functr = 0
			for w in func:
				if w in w2idx:
					if func[w] == 1:
						freql[w2idx[w]] += 1e7
						functr+=1
			
			show_cormat(erpcor, len(w2idx), list(w2idx.keys()), 'erp similarities for top 100 words', save = f'plots/similarity_top100_erp_sorted_{s}.svg', rankby = freql, reverse = True, rankorder = True, div = 73)

		if noise_ceil:
			subrs[s] = erpcor
			if allrs is None:
				allrs = np.copy(erpcor)
			else:
				allrs += erpcor
		
		cmc, pval = stats.spearmanr(erpcor, embcor)
		# cmc = mystats.tau_a(erpcor, embcor)
		# pval = -1
		print('Sub {}: r: {}, p: {}'.format(s, cmc, pval))
		logf.write('{},{},{}\n'.format(s,cmc,pval))

	logf.close()

	if noise_ceil:
		logf = open(noise_ceil.format(datetime.date.today().strftime('%m%d%y')), 'w')
		for s in subs:
			ub, p = stats.spearmanr(subrs[s], allrs/len(subrs))
			lb, p = stats.spearmanr(subrs[s], (allrs - subrs[s]) / (len(subrs) -1))
			# ub = mystats.tau_a(subrs[s], allrs/len(subrs))
			# lb = mystats.tau_a(subrs[s], (allrs - subrs[s]) / (len(subrs) -1))
			logf.write('{},{},{}\n'.format(s, lb, ub))
		logf.close()


def show_cormat(combos, n, labels, title, save = None, rankby = None, reverse = False, rankorder = False, div = None):
	mat = np.ones([n,n])
	ranks = [i for i in range(n)]
	if rankby:
		ranks = sorted(enumerate(rankby), key = lambda x: x[1], reverse = reverse)
		ranks = [i[0] for i in ranks]
		nlabels = [labels[i] for i in ranks]
		labels = nlabels
	if rankorder:
		combos = stats.rankdata(combos)
		maxrank = max(combos) + 1
		for i in range(n):
			mat[i][i] = maxrank
	rmatch = itertools.combinations(ranks, 2)
	for i,(x,y) in enumerate(rmatch):
		mat[x][y] = combos[i]
		mat[y][x] = combos[i]

	# mat = (mat + 1) / 2
	# print(mat.min(), mat.max())

	fig,ax = plt.subplots(figsize = (n*.25, n*.25))
	if rankorder:
		pos = ax.imshow(mat, cmap = plt.cm.Blues)
	else:
		pos = ax.imshow(mat, cmap = plt.cm.RdBu, norm = matplotlib.colors.Normalize(vmin = -1, vmax = 1))

	ax.set_xticks(np.arange(n))
	ax.set_yticks(np.arange(n))
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.colorbar(pos, ax = ax)
	if div != None:
		plt.axvline(div -.5, color = 'red', lw = 3)
		plt.axhline(div -.5, color = 'red', lw = 3)
	ax.set_title(title)
	# fig.tight_layout()

	if not save:
		plt.show()
	else:
		plt.savefig(save, dpi = 300)

	plt.close()



def get_all_n_choose_i(n):
	pcombos = []
	for i in range(1, n+1):
		pcombos += list(itertools.combinations(range(n), i))
	return pcombos


def proc_indiv_semipartials():
	predictors = []
	predictors.append(load_word_embs('top100_embs.txt', True))
	predictors.append(load_word_embs('elmo/elmo_averages.txt', True))
	# mask = load_word_embs('top200_words.txt', False)

	combkeys = set.intersection(*[set(d.keys()) for d in predictors])

	pcombos = get_all_n_choose_i(len(predictors))

	# logf = open('rs_top100_linreg_steps.csv', 'w')
	# logf.write('Sub,' + ','.join(['"'+str(c)+'",' for c in pcombos]) + '\n')
	# logf.write(',' + ','.join(['R^2,coefs']*len(pcombos)) + '\n')

	w2idx = dict()
	freq = load_word_embs('top100_freq.txt', False)
	for i, w in enumerate(freq.keys()):
		w2idx[w] = i

	conc = load_word_embs('all_words_functional.txt', False)
	for w in conc:
		if w in w2idx:
			if conc[w] == 1:
				freq[w] += 1e7

	freq = list(freq.values())
	ids = range(len(w2idx))
	pcors = [corr_mat(emb,emb_cor,ids, w2idx) for emb in predictors]

	erpsum = None
	snum = 0
	for s in subs:
		with open(dataloc.format(s), 'rb') as pin:
			erps = pickle.load(pin)
		erps = remove_average_erp(erps)

		skip = False
		for w in w2idx:
			if w not in erps.keys():
				skip = True
				break
		if skip:
			print(f'Skiping sub {s}, not all words found')
			continue
		snum+=1
		print('Sub {}: Found {} words in common'.format(s, len(w2idx)))

		postbsl = {k: np.take(x, range(PRE, MAXLEN), axis = -1) for k, x in erps.items()}
		erpcor = corr_mat(postbsl, erp_cor, ids, w2idx)
		if erpsum is None:
			erpsum = np.asarray(erpcor)
		else:
			erpsum += erpcor

		# erpcor = stats.rankdata(erpcor)
		# ansdic = linreg_steps(pcors, erpcor, pcombos)
		# print('Sub {}: {}'.format(s, ansdic.values()))
		
		# sprs = mystats.semipartials(pcors[0], pcors[1], erpcor)
		# print('Sub {}: {}'.format(s, sprs))
		
		# logf.write(s+',')
		# for c in pcombos:
		# 	logf.write('{},"{}",'.format(np.sqrt(ansdic[c][0]), ','.join([str(x) for x in ansdic[c][1]])))
		
		# logf.write('{},{}'.format(*sprs))

		# logf.write('\n')

	erpsum /= snum
	r1,p = stats.spearmanr(pcors[0], erpsum)
	r2,p = stats.spearmanr(pcors[1], erpsum)
	print(f'fasttext {r1}, elmo {r2}')
	# show_cormat(erpsum, len(w2idx), list(w2idx.keys()), 'mean erp correlations', rankby = freq, reverse = True, rankorder = True, div = 72, save = 'erp_cormat_sum_rankorder.eps')
	sprs = mystats.semipartials(pcors[0], pcors[1], erpsum)
	print(sprs)

	pcombos = get_all_n_choose_i(len(pcors))
	stepreg = linreg_steps([stats.rankdata(x) for x in pcors], stats.rankdata(erpsum), pcombos)
	print(stepreg)
	
	# logf.close()
	return w2idx, erpsum

def proc_grandavg_reg():
	predictors = []
	predictors.append(load_word_embs('top100_embs.txt', True))
	predictors.append(load_word_embs('elmo/elmo_averages.txt', True))

	with open('gaerp.pickle', 'rb') as pin:
		gaerps = pickle.load(pin)

	w2idx = {w:i for i,w in enumerate(gaerps)}
	for i in range(len(predictors)):
		predictors[i] = corr_mat(predictors[i], emb_cor, range(len(w2idx)), w2idx)

	postbsl = {k: np.take(x, range(PRE, 300), axis = -1) for k, x in gaerps.items()}
	erpcor = corr_mat(postbsl, erp_cor, range(len(w2idx)), w2idx)
	print([stats.spearmanr(erpcor, x) for x in predictors])

	pcombos = get_all_n_choose_i(len(predictors))
	stepreg = linreg_steps([stats.rankdata(x) for x in predictors], stats.rankdata(erpcor), pcombos)
	print(stepreg)

	print(mystats.semipartials(predictors[0], predictors[1], erpcor))


def proc_indiv_partitions_continuous():
	wordembs = load_word_embs(FLAGS.embs, True)
	mask = load_word_embs(FLAGS.mask, False)
	mask = set(mask.keys())
	interval = 25
	length = MAXLEN

	logfr = open(FLAGS.output + '_nobaseline_r_{}.csv'.format(datetime.date.today().strftime('%m%d%y')), 'w')
	logfp = open(FLAGS.output + '_nobaseline_p_{}.csv'.format(datetime.date.today().strftime('%m%d%y')), 'w')
	for s in subs:
		with open(os.path.join(FLAGS.rawdata, 'word_averages_{}.pickle').format(s), 'rb') as pin:
			erps = pickle.load(pin)
		erps = remove_average_erp(erps, baseline_corr = False, partial = (PRE, ONSET))

		w2idx = dict()
		i = 0
		for w in set(erps.keys()) & set(wordembs.keys()) & mask:
			w2idx[w] = i
			i+=1
		print('Sub {}: Found {} words in common'.format(s, len(w2idx)))
		
		ids = [i for i in range(len(w2idx))]
		rvals = []
		pvals = []
		embcor = corr_mat(wordembs, emb_cor, ids, w2idx)
		for seg in range(length - interval):
			erpseg = {k: np.take(x, range(seg, seg + interval), axis = -1) for k, x in erps.items()}
			erpcor = corr_mat(erpseg, erp_cor, ids, w2idx)
			cmc, pval = stats.spearmanr(erpcor, embcor)
			rvals.append(cmc)
			pvals.append(pval)
		print(f'Sub {s} done')
		logfr.write(f'{s},{",".join([str(rvals[i]) for i in range(len(rvals))])}\n')
		logfp.write(f'{s},{",".join([str(pvals[i]) for i in range(len(pvals))])}\n')

	logfr.close()
	logfp.close()

def get_noise_ceilings_continuous():
	mask = load_word_embs(FLAGS.mask, False)
	mask = set(mask.keys())

	interval = 25
	length = MAXLEN

	subrs = dict()
	allrs = None
	for s in subs:
		with open(os.path.join(FLAGS.rawdata, f'word_averages_{s}.pickle'), 'rb') as pin:
			erps = pickle.load(pin)
		erps = remove_average_erp(erps, baseline_corr = True, partial = (100, 150))

		w2idx = dict()
		i = 0
		for w in set(erps.keys()) & mask:
			w2idx[w] = i
			i+=1

		print('Sub {}: Found {} words in common'.format(s, len(w2idx)))

		ids = range(len(w2idx))
		subrs[s] = []
		for seg in range(length - interval):
			erpseg = {k: np.take(x, range(seg, seg + interval), axis = -1) for k, x in erps.items()}
			erpcor = np.array(corr_mat(erpseg, erp_cor, ids, w2idx))
			subrs[s].append(erpcor)

		if allrs is None:
			allrs = copy.deepcopy(subrs[s])
		else:
			for seg in range(length - interval):
				allrs[seg] += subrs[s][seg]

	loglbf = open('noise_ceilings_continuous_lb_{}.csv'.format(datetime.date.today().strftime('%m%d%y')), 'w')
	logubf = open('noise_ceilings_continuous_ub_{}.csv'.format(datetime.date.today().strftime('%m%d%y')), 'w')
	for s in subs:
		ubs = []
		lbs = []
		for seg in range(length - interval):
			ub, p = stats.spearmanr(subrs[s][seg], allrs[seg]/len(subrs))
			lb, p = stats.spearmanr(subrs[s][seg], (allrs[seg] - subrs[s][seg]) / (len(subrs)-1))
			ubs.append(ub)
			lbs.append(lb)
		loglbf.write('{},{}\n'.format(s, ','.join([str(x) for x in lbs])))
		logubf.write('{},{}\n'.format(s, ','.join([str(x) for x in ubs])))
	
	loglbf.close()
	logubf.close()



def proc_grand_avg(savefig = None):
	wordembs = load_word_embs('all_words_embs.txt', True)
	mask = load_word_embs('top100_words.txt', False)
	mask = set(list(mask.keys()))

	gaerps = dict()
	gaerpcor = []
	w2idx = None
	for s in subs:
		with open(dataloc.format(s), 'rb') as pin:
			erps = pickle.load(pin)
		erps = remove_average_erp(erps)
		postbsl = dict()

		for w in set(erps.keys()) & set(wordembs.keys()) & mask:
			if w not in gaerps:
				gaerps[w] = [np.ravel(erps[w]), 1]
			else:
				gaerps[w][0] += np.ravel(erps[w])
				gaerps[w][1] += 1
			postbsl[w] = np.take(erps[w], range(PRE, 300), axis = -1)
		
		if w2idx is None:
			w2idx = dict()
			for i, k in enumerate(gaerps.keys()):
				w2idx[k] = i

		currerpcor = corr_mat(postbsl, erp_cor, range(len(w2idx)), w2idx)
		gaerpcor.append(currerpcor)

	for w in gaerps:
		gaerps[w] = gaerps[w][0] / gaerps[w][1]
	gaerpcor = np.average(gaerpcor, axis = 0)

	print(f'Vocab size is {len(w2idx)}.')
	ids = [i for i in range(len(w2idx))]

	embcor = corr_mat(wordembs, emb_cor, ids, w2idx)

	with open('gaerp.pickle', 'wb') as pout:
		pickle.dump(gaerps, file = pout)

	postbsl = {k: np.take(x, range(PRE, 300), axis = -1) for k, x in gaerps.items()}
	erpcor = corr_mat(postbsl, erp_cor, ids, w2idx)
	smc, pval = stats.spearmanr(erpcor, embcor)
	print(f'Grand average r: {smc}, p: {pval}')
	smc, pval = stats.spearmanr(gaerpcor, embcor)
	print(f'Average of matrices r: {smc}, p: {pval}')

	if savefig:
		freq = load_word_embs('top100_freq.txt', False)
		conc = load_word_embs('all_words_functional.txt', False)
		freql = [freq[w] for w in w2idx]
		functr = 0
		for w in conc:
			if w in w2idx:
				if conc[w] == 1:
					freql[w2idx[w]] += 1e7
					functr+=1
		
		show_cormat(gaerpcor, len(w2idx), list(w2idx.keys()), 'grand average erp similarities for top 100 words', save = savefig, rankby = freql, reverse = True, rankorder = True, div = functr)



def self_corr():
	global w2idx
	n_rep = 100

	with open('top100_words.txt') as fin:
		mask = set(fin.read().rstrip().split('\n'))

	logf = open('selfcorr_top100.csv', 'w')

	for s in subs:
		with open('../../ContinuousStory/raw/EEG_sub_{}.pickle'.format(s), 'rb') as pin:
			data = pickle.load(pin)
		rvals = []
		pvals = []

		for t in range(n_rep):
			sys.stdout.write('\rTesting {}/{}'.format(t+1,n_rep))
			random.seed()

			erp1, erp2 = get_split_erp(data)

			w2idx = dict()
			for i, w in enumerate(set(erp1.keys()) & set(erp2.keys()) & mask):
				w2idx[w] = i
			
			pb1 = {k: np.take(x, range(PRE, MAXLEN), axis = -1) for k, x in erp1.items()}
			pb2 = {k: np.take(x, range(PRE, MAXLEN), axis = -1) for k, x in erp2.items()}
			
			ids = range(len(w2idx))
			cor1 = corr_mat(pb1, erp_cor, ids)
			cor2 = corr_mat(pb2, erp_cor, ids)
			r, p = stats.spearmanr(cor1, cor2)
			rvals.append(r)
			pvals.append(p)
		print()
		r = np.mean(rvals)
		logf.write('{},{},{}\n'.format(s,r,rvals))
		# print(rvals)
		# print(pvals)
		print('Sub {}: average r for {} splits is {}'.format(s,n_rep,r))

	logf.close()


def proc_indiv_permute(trials = 10000):
	# wordembs = load_word_embs('top100_embs.txt')
	wordembs = load_word_embs('elmo/elmo_averages.txt', True)
	mask = load_word_embs('top100_words.txt', False)
	mask = set(list(mask.keys()))

	logf = open('permutation_tests/top100_elmo_perm10000_all_subs.csv', 'w')
	for s in subs:
		sublogf = open('permutation_tests/sub{}_top100_elmo_perm10000.csv'.format(s), 'w')
		with open(dataloc.format(s), 'rb') as pin:
			erps = pickle.load(pin)
		erps = remove_average_erp(erps)

		w2idx = dict()
		for i,w in enumerate(set(erps.keys()) & set(wordembs.keys()) & mask):
			w2idx[w] = i
		print('Sub {}: Found {} words in common'.format(s, len(w2idx)))
		ids = range(len(w2idx))

		postbsl = {k: np.take(x, range(PRE, MAXLEN), axis = -1) for k, x in erps.items()}
		erpcor = corr_mat(postbsl, erp_cor, ids, w2idx)

		# embcor = corr_mat(wordembs, emb_cor, ids, w2idx)
		# cmc, pval = stats.spearmanr(erpcor, embcor)
		# sublogf.write('{},{}\n'.format(cmc,pval))

		r_to_test = cmc
		testperms = []
		for t in range(trials):
			sys.stdout.write('\rTesting {}/{}'.format(t+1,trials))
			randwordembs = scramble_embs(wordembs)

			embcor = corr_mat(randwordembs, emb_cor, ids, w2idx)
			cmc, pval = stats.spearmanr(erpcor, embcor)
			sublogf.write('{},{}\n'.format(cmc,pval))
			testperms.append(cmc)

		testperms = np.asarray(testperms)
		perm_p = sum(testperms>r_to_test)/trials
		print('\rSub {}: Permutaion test p-value: {}'.format(s, perm_p))
		logf.write('{},{},{}\n'.format(s, r_to_test, perm_p))
		sublogf.close()

	logf.close()




def generate_emb_cormat_fig():
	w2idx = dict()
	wordembs = load_word_embs('bert/bert_averages.txt', True)
	freq = load_word_embs('top100_freq.txt', False)
	for i, w in enumerate(freq.keys()):
		w2idx[w] = i

	conc = load_word_embs('all_words_functional.txt', False)
	functr = 0
	for w in conc:
		if w in w2idx:
			if conc[w] == 1:
				freq[w] += 1e7
				functr+=1
	
	freq = list(freq.values())
	cor = corr_mat(wordembs, emb_cor, range(len(w2idx)), w2idx)
	show_cormat(cor, len(w2idx), list(w2idx.keys()), 'bert', rankby = freq, reverse = True, rankorder = True, div = functr, save = 'bert_cormat_top100_cvf.svg')

def get_word_freqs(cols = 6):
	wordfreqs = load_word_embs('top100_words.txt', False)
	wordfreqs = {w:[] for w in wordfreqs}

	for s in subs:
		print(f'Processing sub {s}')
		with open(rawdata_fmt.format(s), 'rb') as pin:
			eegs = pickle.load(pin)
		ctr = collections.Counter()
		for e in eegs:
			ctr.update([e['label']])
		for w in wordfreqs:
			wordfreqs[w].append(ctr[w])

	func = load_word_embs('all_words_functional.txt', False)
	flist = []
	clist = []
	for w, _ in sorted(wordfreqs.items(), key = operator.itemgetter(1), reverse = True):
		if func[w] == 1:
			flist.append(w)
		else:
			clist.append(w)

	ofreq = load_word_embs('all_words_freq_elist.txt', False)
	
	
	def _bundle_stats(word):
		nonlocal wordfreqs, ofreq
		return '{} ({:.2f}, {:.2f}, {})'.format(word, np.mean(wordfreqs[word]), np.std(wordfreqs[word]), int(ofreq[word][0]))
	
	def _build_list(wordlist):
		nonlocal cols
		sout = ''
		for i in range(0, len(wordlist), cols):
			sout += '\t'.join([_bundle_stats(w) for w in wordlist[i:min(len(wordlist), i+cols)]])
			sout += '\n'
		return sout

	sout = 'Function words\n'
	sout += _build_list(flist)
	sout += 'Content words\n'
	sout += _build_list(clist)

	with open('top100_freq_table.tsv', 'w') as fout:
		fout.write(sout)


def get_freq_from_elists():
	elistfmt = '../../ContinuousStory/eventlists/{}_eventlist_with_words.txt'

	counter = collections.Counter()

	with open(elistfmt.format(1)) as fin:
		for line in fin:
			w = line.split()[3]
			w = w.split(',')[0]
			counter.update([w])

	with open('all_words_freq_elist.txt', 'w') as fout:
		for w in counter:
			fout.write(f'{w} {counter[w]}\n')

def get_minfreq_words():
	wlist = load_word_embs('top100_words.txt', False)
	minfreqs = {w:1e7 for w in wlist}
	for s in subs:
		ctr = {w:0 for w in wlist}
		with open(rawdata_fmt.format(s), 'rb') as pin:
			eegs = pickle.load(pin)
		for e in eegs:
			w = e['label']
			if w in ctr:
				ctr[w] += 1
		for w in ctr:
			if ctr[w] < minfreqs[w]:
				minfreqs[w] = ctr[w]

	minfreqs = sorted(minfreqs.items(), key = operator.itemgetter(1), reverse = True)
	with open('top100_minfreq.txt', 'w') as fout:
		for w in minfreqs:
			fout.write(f'{w[0]} {w[1]}\n')


def get_rep_mfs():
	folder = 'rs_elmo_freq_downsampled_mf30_{}'.format(datetime.date.today().strftime('%m%d%y'))
	try:
		os.mkdir(folder)
	except FileExistsError:
		pass

	for i in range(100):
		print(f'Iteration {i+1}/100')
		proc_indiv(log_file = os.path.join(folder, f'elmo_{i}.csv'), equal_sample = 30, noise_ceil = os.path.join(folder, f'noise_ceilings_{i}.csv'))


def main(_):
	if FLAGS.func == 'rsa':
		proc_indiv(log_file = FLAGS.output)
	elif FLAGS.func == 'mw':
		proc_indiv_partitions_continuous()
	elif FLAGS.func == 'plot':
		plot_cormat()
	else:
		# define other functions to run here
		# get_noise_ceilings()
		get_noise_ceilings_continuous()


if __name__ == '__main__':
	flags.DEFINE_string('rawdata', '', 'Folder containing pickled per-subject epoch data', short_name = 'd')
	flags.DEFINE_string('embs', '', 'Embeddings file')
	flags.DEFINE_string('output', '', 'Where to save computed correlations')
	flags.DEFINE_string('mask', '', 'List of words to include')
	flags.DEFINE_enum('func', 'manual', ['manual', 'rsa', 'mw', 'plot'], 'Set which function to run')

	app.run(main)

