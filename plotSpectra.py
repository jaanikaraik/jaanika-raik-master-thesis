#!/usr/bin/python3
import matplotlib.pyplot as plt
import argparse

figureName = ''
parser = argparse.ArgumentParser(description='MCMC.')
parser.add_argument('-n', '--name', default='', help='Length of the chain (each walker).')
args = parser.parse_args()
if args.name != '':
    figureName = str(args.name)

#wavelengthranges
zModel = open('data/zmodel.dat', 'r')
lines = zModel.readlines()
wavelengthRange = lines[5].strip().split()

files = [open('observed.dat', 'r'), open('plotff1', 'r'), open('plotff1i', 'r')]

for file in files:
    x, y = [], []
    for line in file:
        line1 = line.strip().split()
        if float(line1[0]) >= float(wavelengthRange[0]) and float(line1[0]) <= float(wavelengthRange[1]):
            x.append(float(line1[0]))
            y.append(float(line1[1]))
    plt.plot(x,y)

#Either only save or only plot!
#plt.show()
plt.savefig('testedResults/spectrum' + figureName + '.png')
