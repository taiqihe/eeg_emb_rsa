
# Simple bootstrapping procedure to get a confidence interval
import sys
import random
from multiprocessing import Pool

import numpy as np

def b1sample(placeholder):
	global ARR
	s = random.choices(ARR, k = len(ARR))
	return np.mean(s)

def bootstrap(data, trials = 10000):
	global ARR
	ARR = data
	with Pool(8) as p:
		dist = p.map(b1sample, range(trials))
	return dist

def main():
	random.seed()
	data = []
	# Takes a text file consisting of only numbers to be tested, separated by space or tab
	with open(sys.argv[1]) as fin:
		data = fin.read().rstrip().split()
		data = [float(x) for x in data]
	dist = bootstrap(data, 100000)
	print(f'Mean {np.mean(dist)}, 95% CI: {np.percentile(dist, 2.5)}, {np.percentile(dist, 97.5)}')
	

if __name__ == '__main__':
	main()