#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data = pd.read_csv('chain.dat', sep='\s+', names=['walker', 'Vr', 'Teff', 'logg', 'vsini', 'contNorm1', 'contNorm2', 'contNorm3', 'metal', 'loglike'])
data = pd.read_csv('chain.dat', sep='\s+', names=['walker', 'Vr', 'Teff', 'logg', 'vsini', 'metal', 'loglike'])

data.drop('walker',
  axis='columns', inplace=True)
data.drop('loglike',
  axis='columns', inplace=True)

data.iloc[(data.shape[0] // 2):,:]

print(data)

corr = data.corr()
print(corr)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)

#Either only save or only plot!
#plt.show()
plt.savefig('testedResults/correlationHDfixed.png')
#walker Vr Teff logg vsini contNorm contNorm contNorm metal loglike
