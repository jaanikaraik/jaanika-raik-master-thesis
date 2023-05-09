#!/usr/bin/python3
import scipy.stats as stats

mcmc = [3.3140, 2.7448, 0.7233, 0.5225]

chi2 = [3.1025, 2.9269, 0.9343, 0.5209]

print(stats.ttest_rel(mcmc, chi2))
