import sys
import random
from multiprocessing import Pool

import numpy as np

def b1sample(placeholder):
	global ARR
	s = random.choices(ARR, k = len(ARR))
	return np.mean(s)

def worker_init(arr):
	global ARR
	ARR = arr

def bootstrap(data, trials = 10000):
	with Pool(8, initializer = worker_init, initargs = (data,)) as p:
		dist = p.map(b1sample, range(trials))
	return dist

def main():
	random.seed()
	data = []
	with open(sys.argv[1]) as fin:
		data = fin.read().rstrip().split()
		data = [float(x) for x in data]
	dist = bootstrap(data, 100000)
	print(f'Mean {np.mean(dist)}, 95% CI: {np.percentile(dist, 2.5)}\t{np.percentile(dist, 97.5)}')
	

if __name__ == '__main__':
	main()