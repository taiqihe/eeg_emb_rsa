import sys
import random
import math
import pickle
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from scipy import stats

import bootstrapping
import cormat
import mystats


def load_word_embs(file, normalize = True, discard_firstln = False):
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


def fig4():
	# Hard code data from experiment results
	fembr = [0.2336165678,0.05122549169,0.1148548563,0.09658883138,0.08928432256,0.09056991571,0.1025728518,0.1647811144,0.168453011,0.0444622412,0.06384426988,0.03369196246,0.09633191757,0.07980721506,0.08092811871,0.1963774,0.08444332568,0.03772569484,0.09501104213,0.1135930125,0.08977898864,0.03836048248,0.1257111988,0.06884015312,0.06693507318,0.04947700529,0.1263681604,0.09177945368,0.04917699335,0.1722135652,0.08831413768,0.1235870646,0.1657996131,0.08319951439]
	elmor = [0.2752317293,0.04989958443,0.0898714791,0.1286209526,0.168631767,0.08946352001,0.1500439083,0.2128751593,0.1853922492,0.0727970036,0.04698158586,0.0308509667,0.1464941095,0.1523970023,0.07833211229,0.2253534669,0.1330889342,0.05895158768,0.09956857682,0.1630667256,0.1380416392,0.1038128,0.2163806369,0.07981899766,0.1472604033,0.07815214094,0.184129434,0.04930038311,0.0417191323,0.2464440738,0.1154070388,0.09372697709,0.1895647735,0.1065517442]
	fsemir = [0.0972352300648,0.018590026938424,0.076144338792713,0.043296586064995,0.026541243414839,0.037545378394938,0.02062035437531,0.068750433948161,0.071135980564867,0.00845139416329,0.055670074430906,0.026148388541232,0.037528629608641,0.010820559384706,0.037529070468175,0.094735293095352,0.019692660797923,0.019142260371694,0.050972674383046,0.026415606585165,0.036584514900246,-0.017581056841003,0.022570062492831,0.025973187797952,0.007021417255948,0.006263686900747,0.047988919621428,0.064383981348434,0.037769422846524,0.062943283326652,0.025461210423074,0.093068459316512,0.069016181430028,0.028533139141262]
	esemir = [0.155122763056732,0.034538454655286,0.033094164199386,0.073391607641483,0.107493361464858,0.045230698805734,0.107592892574968,0.108919196510965,0.098429654517629,0.048754253935514,0.003735286413452,0.010830130838495,0.086676377454561,0.096894247477698,0.042118833664571,0.106987207435666,0.092618380911071,0.027187657848862,0.042249495459439,0.101520535442719,0.07689790433882,0.080112554719096,0.140162327845717,0.045576598432635,0.104345621004342,0.054427048214846,0.099041058279472,0.001348506980706,0.00857729917817,0.152639354628349,0.067144658601993,0.022811322398248,0.102018246815164,0.065765908503657]
	nceil_up = 0.2912431583
	nceil_low = 0.1978021791

	data = [fembr, elmor, fsemir, esemir]
	means = [np.mean(a) for a in data]
	errors = [bootstrapping.bootstrap(a, 100000) for a in data]
	errors = [[np.percentile(a, 2.5), np.percentile(a, 97.5)] for a in errors]
	for i in range(4):
		errors[i][0] = means[i] - errors[i][0]
		errors[i][1] = errors[i][1] - means[i]

	fig,ax = plt.subplots(figsize = (12,5))
	ax.bar(np.arange(4), means, yerr = np.transpose(errors), alpha = .3)
	ax.bar(np.arange(2), [nceil_up] *2, bottom = [nceil_low]*2, color = 'red', alpha = .3)
	ax.set_ylabel('RSA r')
	ax.set_xticks(np.arange(4))
	ax.set_xticklabels(['Fasttext', 'ELMo', 'Fasttext semipartial', 'ELMo semipartial'])

	Y = []
	X = []
	for i in range(4):
		mi = np.mean(data[i])
		for j in data[i]:
			Y.append(j)
			s = max(.2 - abs(j - mi) * 4, 0.01)
			# print(s)
			X.append(np.random.normal(i, s))
	ax.scatter(X,Y, color = 'green')

	# ax.axhline(0.02389577667, linestyle = '--', alpha = .3)

	fig.tight_layout()
	plt.savefig('figure3.svg', dpi = 300)
	# plt.show()

def load_mw_csv(file, avg = True):
	with open(file) as fin:
		mat = []
		for line in fin:
			mat.append([float(x) for x in line.strip().split(',')[1:]])
	mat = np.asarray(mat)
	if avg:
		mat = np.mean(mat, axis = 0)
	return mat

def continuous_plot():
	elmo = load_mw_csv('rs_top100_mwc_elmo_r_nobaseline_050820.csv')
	fasttext = load_mw_csv('rs_top100_mwc_fasttext_r_nobaseline_050820.csv')
	nceil_low = load_mw_csv('noise_ceilings_continuous_nobaseline_lb_050820.csv')
	nceil_up = load_mw_csv('noise_ceilings_continuous_nobaseline_ub_050820.csv')

	# elmo = load_mw_csv('rs_top100_mwc_elmo_r_042620.csv')
	# fasttext = load_mw_csv('rs_top100_mwc_fasttext_r_042620.csv')
	# nceil_low = load_mw_csv('noise_ceilings_continuous_lb_042620.csv')
	# nceil_up = load_mw_csv('noise_ceilings_continuous_ub_042620.csv')

	time = [i for i in range(-550, 950, 4)]

	fig, ax = plt.subplots(figsize = (10,5))
	l2 = ax.plot(time, elmo, color = 'blue', label = 'ELMo')
	l1 = ax.plot(time, fasttext, color = 'green', label = 'Fasttext')
	nc = ax.fill_between(time, nceil_low, nceil_up, color = 'red', alpha = .3, label = 'Noise Ceiling')
	ax.set_ylabel('Time (ms)')
	ax.set_xlabel('Correlation')
	ax.set_ylim(top = .3)
	ax.set_xlim(left = min(time), right = max(time))

	ax.grid(True)
	ax.axhline(0, linewidth = 1, color = 'black')
	ax.axvline(0, linewidth = 1, color = 'black')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)

	ax.legend()

	fig.tight_layout()
	# plt.show()
	plt.savefig('mwc_nobaseline.svg', dpi = 300)


def continuous_stats():
	# Fill in with actual mw outputs
	elmo = load_mw_csv('', avg = True)
	fasttext = load_mw_csv('', avg = True)
	ncl = load_mw_csv('', avg = True)
	ncu = load_mw_csv('', avg = True)

	time = [i for i in range(-550, 950, 4)]


	# sigids = mystats.fdr_correction(parr, .05)

	fig, ax = plt.subplots(figsize = (10,5))
	l2 = ax.plot(time, elmo, color = 'blue', label = 'ELMo')
	l1 = ax.plot(time, fasttext, color = 'green', label = 'Fasttext')
	nc = ax.fill_between(time, ncl, ncu, color = 'gray', alpha = .3, label = 'Noise Ceiling Range')

	ax.set_ylabel('Time (ms)')
	ax.set_xlabel('Correlation')
	ax.set_ylim(top = .3, bottom = 0)
	ax.set_xlim(left = min(time), right = max(time))

	# ax.grid(True)
	ax.axhline(0, linewidth = 1, color = 'black')
	ax.axvline(0, linewidth = 1, color = 'black')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)

	ax.legend()

	fig.tight_layout()
	plt.show()
	# plt.savefig('rs_mwc_{}.svg'.format(datetime.date.today().strftime('%m%d%y')))



def continuous_stats_sig():
	elmo = load_mw_csv('', avg = False)
	fasttext = load_mw_csv('', avg = False)

	time = [i for i in range(-550, 950, 4)]

	pelmo = []
	pftext = []
	for i in range(len(time)):
		p = mystats.sign_test(np.ravel(elmo[:, i]), 0, alternative = 'greater')
		pelmo.append(p)
		p = mystats.sign_test(np.ravel(fasttext[:, i]), 0, alternative = 'greater')
		pftext.append(p)

	fasttext = np.mean(fasttext, axis = 0)
	elmo = np.mean(elmo, axis = 0)

	sigelmo = mystats.fdr_correction(pelmo, .05)
	sigftext = mystats.fdr_correction(pftext, .05)

	fig, ax = plt.subplots(figsize = (10,5))

	def _plot_sigids(ax, time, sigids, color, at):
		singlepoints = []
		for i in sigids:
			l = (i-1) in sigids
			r = (i+1) in sigids
			if not (l or r):
				singlepoints.append(i)
		# print(singlepoints)
		siglines = np.ma.array([at]*len(time), mask = [False if i in sigids else True for i in range(len(time))])
		sig = ax.plot(time, siglines, lw = 3, color = color)
		ax.plot([time[i] for i in singlepoints], [at]*len(singlepoints), 'o', ms = 3, color = color)

	_plot_sigids(ax, time, sigelmo, 'blue', 2)
	_plot_sigids(ax, time, sigftext, 'green', 1)	

	ax.set_ylabel('Time (ms)')
	ax.set_xlabel('Correlation')
	# ax.set_ylim(top = .3, bottom = 0)
	ax.set_xlim(left = min(time), right = max(time))

	# ax.grid(True)
	ax.axhline(0, linewidth = 1, color = 'black')
	# ax.axvline(0, linewidth = 1, color = 'black')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)

	# ax.legend()

	fig.tight_layout()
	# plt.show()
	plt.savefig('rs_mwc_sig_nobaseline_{}.svg'.format(datetime.date.today().strftime('%m%d%y')))


def small_cormat():
	with open(cormat.dataloc.format(11), 'rb') as pin:
		erps = pickle.load(pin)
	erps = cormat.remove_average_erp(erps)

	embs = cormat.load_word_embs('all_words_embs.txt', True)

	selected_words = ['room', 'from', 'door']
	erps = {x:erps[x] for x in selected_words}
	embs = {x:embs[x] for x in selected_words}

	w2idx = {w:i for i,w in enumerate(selected_words)}

	postbsl = {k: np.take(x, range(cormat.PRE, cormat.MAXLEN), axis = -1) for k, x in erps.items()}
	erpcor = cormat.corr_mat(postbsl, cormat.erp_cor, range(len(w2idx)), w2idx)
	embcor = cormat.corr_mat(embs, cormat.emb_cor, range(len(w2idx)), w2idx)

	cmc, pval = stats.spearmanr(erpcor, embcor)
	print('r: {}, p: {}'.format(cmc, pval))
	print(erpcor, embcor)
	
	cormat.show_cormat(embcor, len(selected_words), selected_words, 'sample similarity matrix', rankorder = False)


if __name__ == '__main__':
	# Select which function to run, some will need to change the data location manually

	# fig4()
	# small_cormat()
	continuous_stats()