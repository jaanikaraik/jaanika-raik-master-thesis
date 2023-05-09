#!/usr/bin/python3
#
#Plot a chain from an MCMC run

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee

#fileName = "chain.dat"
nBurn = -1
figureName = ''

#Optional input files as command line arguments
import argparse
parser = argparse.ArgumentParser(description='Plot a chain from an MCMC run.')
parser.add_argument('file', nargs='?', default='chain.dat', help='Input file with the chain from the MCMC run. Optional, defaults to chain.dat')
parser.add_argument('-b', '--burn', default='', help='Number of MCMC steps to burn, removing them from the final analysis.')
parser.add_argument('-n', '--name', default='', help='Filename of the figures that will be saved.')
args = parser.parse_args()
fileName = args.file
if args.burn != '':
    nBurn = int(args.burn)
if args.name != '':
    figureName = str(args.name)

#Read chain and organize arrays:
#Read the header line
inFile = open(fileName, 'r')
header = inFile.readline()
labels = header.strip('# ').split()
inFile.close()
#Read the main data
chainFile = np.loadtxt(fileName, unpack=True)
nColums = chainFile.shape[0]
nPar = nColums-2

nWalker = chainFile[0,:]
prob = chainFile[-1,:]
pars =  chainFile[1:-1,:]

totWalkers = int(np.max(nWalker))+1

nWalker = nWalker.reshape((-1, totWalkers))
prob = prob.reshape((-1, totWalkers))
nchain = nWalker.shape[0]

parsByWalker = np.zeros((nPar,nchain,totWalkers))
for i in range(nPar):
    #parsByWalker[i,:,:] = chainFile[1+i,:].reshape((-1, totWalkers))
    parsByWalker[i,:,:] = pars[i,:].reshape((-1, totWalkers))

#Some basic stats, for walkers at the end
for i in range(nPar):
    print('end avg ', labels[i], np.average(parsByWalker[i,-1,:]),
          np.std(parsByWalker[i,-1,:]), np.average(prob[-1,:]) )

#Plot chains
for i in range(nPar):
    plt.subplot(nPar,1,i+1)
    plt.ylabel('{:}'.format(labels[i]))
    plt.xlabel('Step number')
    plt.plot(parsByWalker[i,:,:], 'k', alpha=0.1)
    #plt.plot([0,nchain],[0.0, 0.0], 'r')
#plt.show()
plt.savefig('testedResults/walkers' + figureName + '.png')

#Build samples array
#Cut burn-in part of the chain
#The emcee chain has shapes of [nWalkers, nSamples, nParams]
#for the corner.py package, it wants indices of [nSamples (flattned), nParams]
if nBurn < 0: nBurn = nchain//4
print('trimming the first {:} steps'.format(nBurn))
samples = pars[:,totWalkers*nBurn:].T

#Get percentiles for best values and uncertainties
for i in range(nPar):
    par_low, par_mid, par_hig = np.percentile(samples[:,i], [16., 50., 84.])
    par_avg = np.average(samples[:,i])
    par_stdev = np.std(samples[:,i])
    print('{:} = {:.6} +{:.6} / {:.6} (avg {:.6} +/- {:.6})'.format(
        labels[i], par_mid, par_hig-par_mid, par_low-par_mid,
        par_avg, par_stdev) )

#Make a corner plot
import corner
fig = corner.corner(samples, labels=labels,
                      bins=20) #truths=[14., 3.3, 1.0, 0.0, 0.0]

#plt.show()
plt.savefig('testedResults/corner' + figureName + '.png')
