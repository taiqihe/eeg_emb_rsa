import math
from multiprocessing import Pool
import itertools

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def semipartials(X1, X2, Y):
	# take X1 and X2, calculate the semipartial r for each one controlling for the other
	ry1 = stats.pearsonr(X1, Y)[0]
	ry2 = stats.pearsonr(X2, Y)[0]
	r12 = stats.pearsonr(X1, X2)[0]

	sp1 = (ry1 - r12*ry2) / math.sqrt(1-ry2**2)
	sp2 = (ry2 - r12*ry1) / math.sqrt(1-ry1**2)

	return sp1, sp2

def tau_a_comp(arg):
	global ARR
	i,j = arg
	ans = np.sign(ARR[j] - ARR[i])
	return ans[0]*ans[1]

def tau_a(X, Y):
	if len(X)!=len(Y):
		print('Array lengths must agree')

	joint = np.array(list(zip(X,Y)))
	global ARR
	ARR = joint
	with Pool(16) as p:
		ans = p.map(tau_a_comp, itertools.combinations(range(len(joint)), 2))
	return 2*sum(ans) / (len(X)*(len(X)-1))


def sign_test(x, y, alternative = 'two-sided'):
	nmask = sum(x==y)
	signs = x>y
	n = (len(signs) - nmask)
	p = sum(signs)

	if alternative == 'less':
		p = n-p
		alternative = 'greater'

	p = stats.binom_test(p, n, .5, alternative = alternative)
	return p

def fdr_independent(p, alpha):
	# use the Benjamini-Hochberg procedure to control the FDR at level alpha
	# p is a list of p values, returns a list of indices
	
	spid = np.argsort(p)
	for k in range(len(spid)-1, -1, -1):
		if p[spid[k]] <= alpha * (k+1) / len(p):
			break
	return spid[:k+1]


def fdr_correction(plist, alpha):
	# use the Benjamini-Yekutieli procedure to control the FDR at level alpha

	spid = np.argsort(plist)
	cm = sum([1/x for x in range(1, len(plist)+1)])
	for k in range(len(spid)-1, -1, -1):
		if plist[spid[k]] <= alpha * (k+1) / (len(plist) * cm):
			break
	return spid[:k+1]