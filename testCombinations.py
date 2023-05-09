#!/usr/bin/python3

import subprocess


epsilonCoefficients=[1.0, 5.0]
numberOfWalkers=[500, 2000]
walkerSizes=[50, 100]

for includeContnorm in [0,1]:
    for includeMetal in [1]:
        for includeVmic in [0]:
            for epsilonCoefficient in epsilonCoefficients:
                for num in numberOfWalkers:

                    for size in walkerSizes:
                        print(includeContnorm)
                        name = "_hd_5100_5200_" + str(includeContnorm)  + "_" + str(includeMetal) + "_" + str(includeVmic)  + "_" + str(epsilonCoefficient)   + "_" + str(num)  + "_" + str(size)
                        subprocess.run(['./zemceeWrap03cont.py',
                                        '--contnorm', str(includeContnorm),
                                        '--metal', str(includeMetal),
                                        '--vmic', str(includeVmic),
                                        '--epsilons',str(epsilonCoefficient),
                                        '--numberofwalkers', str(num),
                                        '--chainsize', str(size),
                                        '--name', name])
                        #textfile should include real and user time
                        subprocess.run(['./plotSpectra.py','--name', name])
                        burnin = size // 2
                        subprocess.run(['./plotChain.py','--name', name,'--burn', str(burnin)])
