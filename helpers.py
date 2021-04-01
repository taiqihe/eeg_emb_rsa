import os

from scipy import stats
import numpy as np

def csvreader(fn):
	with open(fn) as fin:
		data = []
		for line in fin:
			data.append(line.strip().split(','))
	return data


def get_multifile_averages(pref, maxnum, pos = 1, out = 'averages.csv'):
	agg = []
	for i in range(maxnum):
		fn = pref.format(i)
		if os.path.exists(fn):
			data = csvreader(fn)
			if len(agg) == 0:
				for ln in range(len(data)):
					agg.append([data[ln][pos]])
			else:
				for ln in range(len(data)):
					agg[ln].append(data[ln][pos])
		else:
			print(f'{fn} not found.')
	
	with open(out, 'w') as fout:
		for l in agg:
			fout.write(','.join(l) + '\n')


def get_wilcoxon_single(file):
	data = csvreader(file)


def read_simlex(file):
	with open(file) as fin:
		fin.readline()
		relations = []
		for line in fin:
			curr = line.strip().split()
			relations.append(((curr[0], curr[1]), float(curr[3])))
	return relations

def get_simlex_correlations(embsdict, embscor, simlex_file = 'SimLex-999.txt'):
	simlex = read_simlex(simlex_file)
	embsmatched = []
	simlexcor = []
	counter = 0
	for w,c in simlex:
		if w[0] in embsdict and w[1] in embsdict:
			counter += 1
			embsmatched.append(embscor(embsdict[w[0]], embsdict[w[1]]))
			simlexcor.append(c)
	print(f'Found {counter} pairs in common with SimLex-999.')
	return stats.spearmanr(simlexcor, embsmatched)
	

def test_sub_stats(data):
	r1 = stats.spearmanr(np.ravel(data['room']), np.ravel(data['from']))[0]
	r2 = stats.spearmanr(np.ravel(data['room']), np.ravel(data['door']))[0]
	r3 = stats.spearmanr(np.ravel(data['door']), np.ravel(data['from']))[0]
	print(r1, r2, r3)
	

def main():
	get_multifile_averages('rs_elmo_freq_downsampled_mf30_050820/elmo_{}.csv', 100, pos = 1, out = 'rs_elmo_mf30x100_agg.csv')


if __name__ == '__main__':
	main()