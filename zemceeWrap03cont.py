#!/usr/bin/python3

#A python wrapper around ZEEMAN/LMA, with the MCMC paramter estimator emcee

import numpy as np
import subprocess
import emcee

import time
import argparse

#Currently supported parameter keywords
knownPars = ('Vr', 'vsini', 'Vmic', 'Vmac', 'Teff', 'logg', 'metal', 'contFlx','Bmono','FFmono','Bdip', 'FFdip', 'element', 'abun', 'contNorm')

#Modify the main input file for Zeeman, inlmam.dat
#Read the file into memory, modify any parameters,
#then write a new version back to disk.
def updateInlma(freePar, freeParID, fixedParams):
    """
    Read in and then re-write the inlmam.dat file.
    This takes the free parameters (and their keywords) from freePar and freeParID
    and fixed parameters (as a dict) in fixedParams.
    The inlmam.dat file must already exist and be in the correct format
    for Zeeman. All parameters will be set to fixed.
    Parameter values from inlmam.dat will be reused unless they are specified
    by the input function arguments.
    """
    #Read the Zeeman-LMA input
    fInlma = open('inlmam.dat', 'r')
    
    txtInlma = []
    monoB = []
    monoff = []
    numMono = 0
    dipB = []
    dipff=[]
    numDip = 0
    numEl = 0
    listEl = []
    abuns = []
    stopEl = 0
    i=1
    for line in fInlma: #Probably possible to optimize!!!!
        if i == 4:
            vr = float(line.split()[0])
        if i == 6:
            vsini = float(line.split()[0])
        if i == 8:
            vmic = float(line.split()[0])
        if i == 10:
            vmac = float(line.split()[0])
        if i == 12:
            Teff = float(line.split()[0])
        if i == 14:
            logg = float(line.split()[0])
        if i == 16:
            metal = float(line.split()[0])
        if i == 18:
            contFlux = float(line.split()[0])
        if i == 20:
            numMono = int(line.split()[0])
        if (i > 21) and (i < 22+numMono):
            monoB += [float(line.split()[0])]
            monoff += [float(line.split()[1])]
        if i == 23 + numMono:
            numDip = int(line.split()[0])
        if (i > 24+numMono) and (i < 25+numMono+numDip):
            dipB += [float(line.split()[0])]
            dipff += [float(line.split()[1])]
        if (i > 25+numMono+numDip):
            if (stopEl == 0):
                if (line.strip()[0] == '='):
                    stopEl = 1
                else:
                    numEl += 1
                    listEl += [int(line.split()[0])]
                    abuns += [float(line.split()[1])]
        
        txtInlma += [line]
        i += 1
    
    fInlma.close()
    if(stopEl == 0): print('ERROR missing end of inlmam.dat (missing ending =)')

    
    #Set the new/modified parameters, overwriting the values read from file.
    iQuit = False
    if(len(freePar) != len(freeParID)):
        print('ERROR: missmatch in freePar and freeParID lengths(2)')
        iQuit = True
    if 'Bmono' in freeParID: monoB = []
    if 'FFmono' in freeParID: monoff = []
    if 'Bdip' in freeParID: dipB = []
    if 'FFdip' in freeParID: dipff = []
    if ('abun' in freeParID) or ('abun' in fixedParams): abuns = []
    for i in range(len(freePar)):
        if freeParID[i] == 'Vr':      vr       = freePar[i]
        if freeParID[i] == 'vsini':   vsini    = freePar[i]
        if freeParID[i] == 'Vmic':    vmic     = freePar[i]
        if freeParID[i] == 'Vmac':    vmac     = freePar[i]
        if freeParID[i] == 'Teff':    Teff     = freePar[i]
        if freeParID[i] == 'logg':    logg     = freePar[i]
        if freeParID[i] == 'metal':   metal    = freePar[i]
        if freeParID[i] == 'contFlx': contFlux = freePar[i]
        if freeParID[i] == 'Bmono':   monoB   += [freePar[i]]
        if freeParID[i] == 'FFmono':  monoff  += [freePar[i]]
        if freeParID[i] == 'Bdip':    dipB    += [freePar[i]]
        if freeParID[i] == 'FFdip':   dipff   += [freePar[i]]
        #if freeParID[i] == 'element': #Element should never be free!
        if freeParID[i] == 'abun':    abuns   += [freePar[i]]
    
    for fixID, fixPar in fixedParams.items():
        if fixID == 'Vr':      vr       = fixPar
        if fixID == 'vsini':   vsini    = fixPar
        if fixID == 'Vmic':    vmic     = fixPar
        if fixID == 'Vmac':    vmac     = fixPar
        if fixID == 'Teff':    Teff     = fixPar
        if fixID == 'logg':    logg     = fixPar
        if fixID == 'metal':   metal    = fixPar
        if fixID == 'contFlx': contFlux = fixPar
        if fixID == 'Bmono':   monoB    = fixPar
        if fixID == 'FFmono':  monoff   = fixPar
        if fixID == 'Bdip':    dipB     = fixPar
        if fixID == 'FFdip':   dipff    = fixPar
        if fixID == 'element': listEl   = fixPar
        if fixID == 'abun':    abuns   += fixPar

    #Extra error trapping (make sure things are lists and have the right length)
    if isinstance(monoB, (int, float)):  monoB  = [monoB]
    if isinstance(monoff, (int, float)): monoff = [monoff]
    if isinstance(dipB, (int, float)):   dipB   = [dipB]
    if isinstance(dipff, (int, float)):  dipff  = [dipff]
    if isinstance(listEl, (int, float)): listEl = [listEl]
    if isinstance(abuns, (int, float)):  abuns  = [abuns]
    if(len(monoB) != len(monoff)):
        print('ERROR inconsistent Bmono and FFmono lists:', monoB, monoff)
        iQuit = True
    if(len(dipB) != len(dipff)):
        print('ERROR inconsistent Bdip and FFdip lists:', dipB, dipff)
        iQuit = True
    if(len(listEl) != len(abuns)):
        print('ERROR inconsistent element and abundance lists:', listEl, abuns)
        iQuit = True
    if iQuit:
        print('\n')
        import sys
        sys.exit()        
    
    #Write the Zeeman-LMA input
    fInlma = open('inlmam.dat', 'w')
    i = 1
    j = 0
    for line in txtInlma:
        
        if i == 4:
            fInlma.write('{:14.8e}  {:1n}\n'.format(vr, 0))
        elif i == 6:
            fInlma.write('{:14.8e}  {:1n}\n'.format(vsini, 0))
        elif i == 8:
            fInlma.write('{:14.8e}  {:1n}\n'.format(vmic, 0))
        elif i == 10:
            fInlma.write('{:14.8e}  {:1n}\n'.format(vmac, 0))
        elif i == 12:
            fInlma.write('{:10.4f}      {:1n}\n'.format(Teff, 0))
        elif i == 14:
            fInlma.write('{:7.5f}         {:1n}\n'.format(logg, 0))
        elif i == 16:
            fInlma.write('{:8.5f}        {:1n}\n'.format(metal, 0))
        elif i == 18:
            fInlma.write('{:8.5f}        {:1n}\n'.format(contFlux, 0))
        elif i == 20:
            fInlma.write('{:2n}\n'.format(len(monoB)))
            j = 0
        elif i == 21:
            #Write the info header line and the new Bmono lines
            fInlma.write(line)
            for k in range(len(monoB)):
                fInlma.write('{:9.3f}    {:16.10e}   {:1n}       {:1n}\n'.format(
                    monoB[k], monoff[k], 0, 0))
                #fInlma.write('{:9.3f}    {:9.7f}   {:1n}       {:1n}\n'.format(
                #    monoB[k], monoff[k], 0, 0))
        elif (i > 21) and (i < 22+numMono):
            #Do nothing for the old Bmono list
            j += 1
        elif i == 23 + numMono:
            fInlma.write('{:2n}\n'.format(len(dipB)))
            j = 0
        elif i == 24 + numMono:
            #Write the info header line and the new Bdip lines
            fInlma.write(line)
            for k in range(len(dipB)):
                fInlma.write('{:9.3f}    {:16.10e}   {:1n}       {:1n}\n'.format(
                    dipB[k], dipff[k], 0, 0))
                #fInlma.write('{:9.3f}    {:9.7f}   {:1n}       {:1n}\n'.format(
                #    dipB[k], dipff[k], 0, 0))
        elif (i > 24+numMono) and (i < 25+numMono+numDip):
            #Do nothing for the old Bdip list
            j += 1
        elif i == 25+numMono+numDip:
            #Write the info header line and the new abundance lines
            fInlma.write(line)
            for k in range(len(listEl)):
                fInlma.write('{:<2n}       {:8.5f}  {:1n}\n'.format(
                    listEl[k], abuns[k], 0))
        elif (i > 25+numMono+numDip) and (i < 26+numMono+numDip+numEl):
            j += 1
        else:
            fInlma.write(line)
            
        i += 1
    fInlma.close()
    return


def getZeemanSpec(vr, modelName = 'plotff1'):
    """
    Read in the output spectrum from Zeeman.
    Apply a Doppler shift here.  Interpolating onto a grid of observed points
    should be in the calling routine.
    """
    #wl, specI, specQ, specU, specV = np.loadtxt('plotff1i', unpack=True)
    #wl = [float(line.split()[0]) for line in open(modelName, 'r')]
    #specI = [float(line.split()[1]) for line in open(modelName, 'r')]
    wl = []
    specI = []
    fIn = open(modelName, 'r')
    for line in fIn:
        vals = line.split()
        wl.append(float(vals[0]))
        specI.append(float(vals[1]))
    fIn.close()
    
    #Apply the Doppler shift correction to the synthetic spectrum
    wla = np.array(wl)
    wla = wla + wla*vr/2.99792458e10
    return wla, np.array(specI)

def getObsSpecInRange(wlRanges, obsName = 'observed.dat'): 
    """
    Read in the observation file for fitting/calculating likelihoods,
    Then return only regions of it in the wavelength ranges specified by wlRanges.
    wlRanges should be a 2D array, with the 1st dimension being different wavelength ranges,
    and the second dimension being a start and end wavelength.
    """
    wlObsFull, specIobsFull, errObsFull = getObsSpec(obsName)
    #Sort by wavelength (later interpolation functions may need data sorted by wavelength)
    try:
        #This seems to be more efficient for spectral points which are mostly already sorted
        isort = np.argsort(wlObsFull, kind='stable')
    except ValueError:
        #But any sort algorithm should be good enough!
        isort = np.argsort(wlObsFull)
    wlObsFull = wlObsFull[isort]
    specIobsFull = specIobsFull[isort]
    errObsFull = errObsFull[isort]
    #Get just the portions of the observation to fit
    indObs = getObsInWlRange(wlObsFull, wlRanges)
    wlObs = wlObsFull[indObs]
    specIobs = specIobsFull[indObs]
    errObs = errObsFull[indObs]
    return wlObs, specIobs, errObs

def getObsSpec(obsName = 'observed.dat'):
    """
    Read in the observation file for fitting/calculating likelihoods.
    """
    #wl, specI, err = np.loadtxt(obsName, unpack=True, usecols = (0,1,5))
    wl = []
    specI = []
    err = []
    fIn = open(obsName, 'r')
    line = fIn.readline()
    ncols = len(line.split())
    fIn.seek(0)
    for line in fIn:
        vals = line.split()
        #For 6 column polarimetric LibreESPRIT .s format
        intensity = float(vals[1])
        if intensity >= 0.0:
            if ncols == 6:
                wl.append(float(vals[0]))
                specI.append(intensity)
                err.append(float(vals[5]))
            #For 3 column intensity .s format
            elif ncols == 3:
                wl.append(float(vals[0]))
                specI.append(intensity)
                err.append(float(vals[2]))
            else:
                raise ValueError('Got unexpected number of columns when reading observation file {:}, {:}'.format(obsName, ncols))
    fIn.close()
    return np.array(wl), np.array(specI), np.array(err)


def getObsInWlRange(wlObs, wlRanges):
    """
    For each start and end pair of wavelengths, get the parts of the observation in that range.
    wlRanges should be a 2D array, with the 1st dimension being different wavelength ranges,
    and the second dimension being a start and end wavelength.
    """
    indUseAll = np.zeros_like(wlObs, dtype=bool)
    for i in range(wlRanges.shape[0]):
        indUseAll += (wlObs >= wlRanges[i,0]) & (wlObs <= wlRanges[i,1])
    #Sanity check for wavelength ranges
    if np.all(np.logical_not(indUseAll)):
        print('ERROR: no observed points in requested wavelength range!')
    return indUseAll


#Data likelyhood P(y|x,sigma,params)
#Calls the main modelling program
def lnlike(freePar, freeParID, fixedParams, wlObs, specIobs, errObs, resultFile=None, verbose=False):
    """
    Calculate the log likelyhood nl(P(y|x,sigma,params)),
    this runs the main Zeeman code, and uses an observation to calculate chi^2.
    Takes the free parameters, keywords for the free parameters,
    as well as a dict for the fixed parameters, all passed in to Zeeman.
    Parameters not specified as free or fixed should be read from the standard
    Zeeman input files.
    """
    
    #Calculate the model spectrum for this set of parameters with Zeeman
    updateInlma(freePar, freeParID, fixedParams)
    if verbose: print("Running Zeeman")
    subprocess.call("./lmamp")

    #Include a Doppler shift for radial velocity
    if 'Vr' in freeParID:
        vr = freePar[freeParID.index('Vr')]
    elif 'Vr' in fixedParams:
        vr = fixedParams['Vr']
    else:
        vr = 0.0

    #Read the output from Zeeman
    wlSyn, specIsyn = getZeemanSpec(vr)

    #Include an extra continuum normalization if necessary
    contPoly = np.zeros_like(wlSyn)
    wlRef =  0.5*(wlSyn[0]+wlSyn[-1])
    if 'contNorm' in freeParID:
        nContPol = 0
        for i, name in enumerate(freeParID):
            if name == 'contNorm':
                contPoly += freePar[i]*(wlSyn-wlRef)**nContPol
                nContPol += 1
        specIsyn = specIsyn / contPoly
    elif 'contNorm' in fixedParams:
        polyVals = fixedParams['contNorm']
        for i, val in enumerate(polyVals):
            contPoly += val*(wlSyn-wlRef)**i
        specIsyn = specIsyn / contPoly

    #Interpolate the model onto the observation
    synIinterp = np.interp(wlObs, wlSyn, specIsyn)
    #if verbose:
    if True:
        fOuti = open('outSpeci.dat','w')
        for i in range(wlObs.size):
            fOuti.write('{:5.4f} {:13.10f}\n'.format(wlObs[i], synIinterp[i]))
        fOuti.close()
    
    chi2 = np.sum(((specIobs - synIinterp)/errObs)**2)
    if verbose:
        print('chi2', chi2, 'reduced chi2', chi2/(wlObs.size-len(freePar))) #write this to file!
        resultFile.write('chi2 ' + str(chi2) + '\n' + 'reduced chi2 ' + str(chi2/(wlObs.size-len(freePar))) + '\n')
    lnlike = -0.5*chi2 
    #lnlike = -0.5*chi2 + np.sum(np.log(np.sqrt(1./(2.*np.pi*sigma**2))))

    return lnlike


#Prior probability P(params)
def lnprior(freePar, freeParID):
    """
    Calculate the priors for the parameters ln(P(params)), return -inf for invalid values.
    Takes a list of parameter values (params), and parameter keyword names (freeParID)
    """
    if(len(freePar) != len(freeParID)): print('missmatch in freePar and freeParID lengths')

    ffsum = 0.0
    #Require parameters to be in valid ranges
    for i in range(len(freePar)):
        if freeParID[i] == 'vsini':
            if freePar[i] < 0.0:
                return -np.inf
        if freeParID[i] == 'Vmic':
            if freePar[i] < 0.0:
                return -np.inf
        if freeParID[i] == 'Vmac':
            if freePar[i] < 0.0:
                return -np.inf
        if freeParID[i] == 'Teff':
            if freePar[i] > 30000.0 or freePar[i] < 3500.0:
                return -np.inf
        if freeParID[i] == 'logg':
            if freePar[i] > 5.0 or freePar[i] < 2.5:
                return -np.inf
        if freeParID[i] == 'contFlx':
            if freePar[i] < 0.0:
                return -np.inf
        
        if freeParID[i] == 'FFmono':
            if freePar[i] < 0.0:
                return -np.inf
            #Find some way to enforce filling factors summing to <= 1.0?
            #It could suffice to just reject those models.
            ffsum += freePar[i]
            if ffsum > 1.0:
                return -np.inf
        if freeParID[i] == 'FFdip':
            if freePar[i] < 0.0:
                return -np.inf
            #Reject models with a total filling factor > 1
            ffsum += freePar[i]
            if ffsum > 1.0:
                return -np.inf

        if freeParID[i] == 'abun':
            if freePar[i] > -0.5 or freePar[i] < -20.0:
                return -np.inf
    
    return 0.0

 
#Full posterior probability P(params|x,y,sigma) ~ P(params)*P(y|x,sigma,params)
def lnprob(freePar, freeParID, fixedParams, wlObs, specIobs, errObs):
    """
    Calculate the full posterior probability P(params|x,y,sigma)
    as ln(P(params|x,y,sigma)) = ln(P(params)) + ln(P(y|x,sigma,params)).
    This relies on the lnprior() and lnlike() functions.
    """
    lnpri = lnprior(freePar, freeParID)
    #If these parameters are bad reject them before running the model.
    if not np.isfinite(lnpri):
        return -np.inf
    lnlik = lnlike(freePar, freeParID, fixedParams, wlObs, specIobs, errObs)
    #If these parameters are off the grid, reject the point too.
    if not np.isfinite(lnlik):
        return -np.inf
    return lnpri + lnlik



def parseInputParams(params, epsilons, fixedParams):
    """
    Conver the input dicts of parameter keywords and values into a list of parameters
    for emcee and a list of parameter keywords/names for other routines later.
    Do this for the free parameter values ('params'), and the initial dispersions
    of free parameters ('epsions').
    """
    #Clean up an possible messiness in the fixedParams dict
    #(mostly things that could be lists or single values: make them all lists)
    for key, value in fixedParams.items():
        if key in ('element','abun','Bmono','FFmono','Bdip', 'FFdip', 'contNorm'):
            if isinstance(value, (int, float)):
                fixedParams[key] = [value]
    
    #Build lists of the free parameter values, free parameter keyword IDs,
    #and dispersion for the initial free parameter distribution.
    freePar = []
    freeParID = []
    freeParEps = []
    print('Using free fitting parameters with initial values:')
    for key, value in params.items():
        if value != None:
            if key in epsilons:
                #If the value exists and there is a corresponding epsilon
                print(' ', key, value, '+/-', epsilons[key])
                if isinstance(value, (int, float, complex)):
                    freePar += [value]
                    freeParID += [key]
                    freeParEps += [epsilons[key]]
                elif isinstance(value, (list, tuple)):
                    for val in value:
                        freePar += [val]
                        freeParID += [key]
                    for val in epsilons[key]:
                        freeParEps += [val]
                else:
                    print('ERROR unsupported data type in dict of free parameters', key, value)
                    import sys
                    sys.exit()
            else:
                #If there is no correspoinding epsilon
                print('ERROR missing epsilon for free value:', key, value)
                import sys
                sys.exit()
                
            #Make sure there are atomic numbers for each abundance given
            if key == 'abun':
                if isinstance(value, (int, float)): value = [value]
                nfixedAbuns = 0
                if 'abun' in fixedParams: nfixedAbuns = len(fixedParams['abun'])
                if len(value) + nfixedAbuns != len(fixedParams['element']):
                    import sys
                    sys.exit('ERROR: Inconsistent number of atomic numbers and element abundances')

            #Make sure there are the right number of magnetic field strengths if using filling factors
            if key == 'FFmono':
                if ('Bmono' in params):
                    if len(value) != len(params['Bmono']):
                        import sys
                        sys.exit('ERROR: Inconsistent number of Bmono and FFmono values')
                elif ('Bmono' in fixedParams):
                    if len(value) != len(fixedParams['Bmono']):
                        import sys
                        sys.exit('ERROR: Inconsistent number of Bmono and FFmono values')
                else:
                    import sys
                    sys.exit('ERROR: Missing list of Bmono')
            if key == 'FFdip':
                if ('Bdip' in params):
                    if len(value) != len(params['Bdip']):
                        import sys
                        sys.exit('ERROR: Inconsistent number of Bdip and FFdip values')
                elif ('Bdip' in fixedParams):
                    if len(value) != len(fixedParams['Bdip']):
                        import sys
                        sys.exit('ERROR: Inconsistent number of Bdip and FFdip values')
                else:
                    import sys
                    sys.exit('ERROR: Missing list of Bdip')

    if len(freePar) != len(freeParEps):
        raise ValueError ('mismatch in number of free parameter initial values'\
            +' and epsilons: {:} != {:}'.format(len(freePar), len(freeParEps)))
    return freePar, freeParID, freeParEps


def runMCMC(nchain, nburn, nwalkers, params, epsilons, fixedParams, wlRanges, resultName, verbose=False):
    """
    Run the main MCMC routine, calling emcee.
    """

    #EDIT: measure time consumption
    start_time = time.time()
    user_time_start = time.perf_counter()

    freePar, freeParID, freeParEps = parseInputParams(params, epsilons, fixedParams)

    #Read in the observation, and get the pixels in the range(s) to be fit
    wlObs, specIobs, errObs = getObsSpecInRange(wlRanges)
    
    #Setup input for emcee
    ndim = len(freePar)
    #Set random initial walker positions for emcee
    pos0 = [freePar + freeParEps*np.random.randn(ndim) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(freeParID, fixedParams, wlObs, specIobs, errObs))
    
    #Full MCMC run
    #outpos, outlnprob, outrngstate =  sampler.run_mcmc(pos, nchain)
    #Or iterate through saving results
    foutChain = open("chain.dat", "w")
    foutChain.write('#'+(' '.join(freeParID))+'\n')
    foutChain.close()
    iStep = 0
    for state in sampler.sample(pos0, iterations=nchain, store=True):
        position, lnprobability, rng = state
        ##position = sampler.get_chain()[-1]  #equivalently
        ##lnprobability = sampler.get_log_prob()[-1]  #equivalently
        foutChain = open("chain.dat", "a")
        for k in range(position.shape[0]): #loop over walkers at this step
            str_values = ""
            for value in position[k]:
                str_values += "{:16e} ".format(value)
            foutChain.write("{:4d} {:s} {:}\n".format(k, str_values, lnprobability[k]))
            if verbose: print("{:n} {:4d} {:s} {:}".format(iStep, k, str_values, lnprobability[k]))
        #print("{:},".format(iStep),end=" ")
        #EDIT: taskbar
        print("{:}%  READY".format(iStep/nchain*100))
        print("step {:}".format(iStep+1))
        foutChain.close()
        iStep += 1
    #Save final walker positions
    outpos = position
    outlnprob = lnprobability
    ##outpos = sampler.get_chain() #full chain of positions (all steps, all walkers, all param)
    ##outlnprob = sampler.get_log_prob() #full chane of ln prob (all steps, all walkers)
    
    #Print output about the final state
    for i in range(ndim):
        print('ending par ({:}) avg {:} std {:} lnP {:}'.format(
            freeParID[i], np.average(outpos[:,i]), 
            np.std(outpos[:,i]), np.average(outlnprob) ))
    print('acceptance fraction {:}'.format(np.average(sampler.acceptance_fraction)))
    
    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate,
    # even if it isn't trustworthy.
    tau = sampler.get_autocorr_time(tol=0)
    print('autocorrelation time estimate {:}'.format(tau))
    
    
    #A quick 'best parameter value' analysis
    #Cut burn-in part of the chain
    if nburn < nchain and nburn >= 0:
        samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    else:
        samples = sampler.chain[:, :, :].reshape((-1, ndim))
    
    #Get final best values, using central and 1sigma values from percentiles
    finalParams = []
    for i in range(ndim):
        par_low, par_mid, par_hig = np.percentile(samples[:,i], [16., 50., 84.])
        finalParams += [par_mid]
    
    #EDIT: create a file
    resultFile = open('testedResults/print' + resultName + '.txt', 'x')

    #Save a final model spectrum, with the above 'best' parameters
    lnlik = lnlike(finalParams, freeParID, fixedParams, wlObs, specIobs, errObs, resultFile, verbose=True)

    #Print central and 1sigma values from percentiles
    finalParams = []
    indFF = 0
    indAbun = 0
    indCont = 0
    for i in range(ndim):
        par_low, par_mid, par_hig = np.percentile(samples[:,i], [16., 50., 84.])
        if freeParID[i] == 'FFmono':
            pparName = '{:}_{:}'.format(freeParID[i], fixedParams['Bmono'][indFF])
            indFF += 1
        elif freeParID[i] == 'abun':
            pparName = '{:}_{:}'.format(freeParID[i], fixedParams['element'][indAbun])
            indAbun += 1
        elif freeParID[i] == 'contNorm':
            pparName = '{:}_{:}'.format(freeParID[i], indCont)
            indCont += 1
        else:
            pparName = freeParID[i]
        printOut = '{:}     {:.6} {:+.6} {:+.6}'.format(pparName, par_mid,
                                            par_hig-par_mid, par_low-par_mid)
        print(printOut)
        resultFile.write(printOut + '\n')


    #EDIT:
    print('final diagnostics:')
    acceptanceFraction = 'acceptance fraction {:}'.format(np.average(sampler.acceptance_fraction))
    print(acceptanceFraction)
    resultFile.write(acceptanceFraction + '\n')

    tau = sampler.get_autocorr_time(tol=0) #tol=0
    autocorrelationTime = 'autocorrelation time estimate {:}'.format(tau)
    print(autocorrelationTime)
    resultFile.write(autocorrelationTime + '\n') # + '\n'

    #EDIT: measure time consumption
    end_time = time.time()
    real_time = end_time - start_time
    resultFile.write('real: {:}\n'.format(real_time))
    print(time.perf_counter()-user_time_start)
    
    return sampler.chain


def linux_time_to_seconds(time_str):
    minutes, seconds = map(float, time_str.replace(',', '.').split('m'))
    return minutes * 60 + seconds

def prepareParams():
    #### stellar parameters ####
    #Know dictionary keywords are:
    # Vr,  vsini,  Vmic,  Vmac,  Teff,  logg,  metal,  contFlx,
    # Bmono (provide a list), FFmono (list), Bdip (list), FFdip (list),
    # element (list, allways fixed), abun (list), contNorm (list)

    #'metal':0.0, 'metal':0.1,
    #params =   {'Vr':30.0e5, 'Teff':15000., 'logg':3.4, 'vsini':65.0e5, 'contNorm':[1.0, 0.0, 0.0] }
    params =   {'Vr':30.0e5, 'Teff':10000., 'logg':3.4, 'vsini':65.0e5}
    params = {'Vr':30.0e5, 'Teff':15000., 'logg':3.4, 'vsini':65.0e5}
    epsilons = {'Vr':0.1e5, 'Teff':100., 'logg':0.1, 'vsini':1.0e5} #, 'contNorm':[0.01, 0.001, 0.001]
    fixedParams = {}


    #EDIT: read the wavelength range directly from file
    zModel = open('data/zmodel.dat', 'r')
    lines = zModel.readlines()
    wavelengthRange = lines[5].strip().split()
    line = [float(wavelengthRange[0]), float(wavelengthRange[1])]
    wlRanges = np.array([line])

    #params defines initial parameters passed to the fitting routine.
    #Omitted values will be read from the input file (inlmam.dat).
    #epsilons defines the initial (Gaussian) distribution of parameters
    #for the ensemble of walkers in the MCMC routine.
    #For a parameter to be 'fitable' (free in this modelling) it needs a
    #value in the params dict and in the epsilons dict.
    #For a parameter to be 'fixed' it can be set it in the fixedParams dict,
    #or it can be omitted here and set in the usual Zeeman input files.

    #### emcee parameters ####

    #Optional input files as command line arguments
    nchain = 50
    nwalkers = 25
    resultName = ''
    epsilonsCoefficient = 1.0
    includeMetal = True
    includeContnorm = True
    includeVmic = False

    parser = argparse.ArgumentParser(description='MCMC.')
    parser.add_argument('-c', '--chainsize', default=50, help='Length of the chain (each walker).')
    parser.add_argument('-w', '--numberofwalkers', default=25, help='Number of walkers running in parallel.')
    parser.add_argument('-n', '--name', default='', help='Name of the file where the final results will be printed out.')
    #new parameters
    parser.add_argument('-e', '--epsilons', default=1.0, help='Multiplying constant of epsilon values.')
    parser.add_argument('-m', '--metal', default=True, help='If true, incude metallicity as a free parameter.')
    parser.add_argument('-x', '--contnorm', default=True, help='If true, contnorm is free parameter, if false, then it is fixed.')
    parser.add_argument('-v', '--vmic', default=False, help='If true, Vmic is free parameter, if false, then it is fixed.')

    args = parser.parse_args()
    if args.chainsize != '':
        nchain = int(args.chainsize)
    if args.numberofwalkers != '':
        numberofwalkers = int(args.numberofwalkers)
    if args.name != '':
        resultName = str(args.name)
    if args.epsilons != '':
        epsilonsCoefficient = float(args.epsilons)
    if args.metal != '':
        includeMetal = bool(int(args.metal))
    if args.contnorm != '':
        includeContnorm = bool(int(args.contnorm))
    if args.vmic != '':
        includeVmic = bool(int(args.vmic))


    if includeContnorm:
        params['contNorm'] = [1.0, 0.0, 0.0]
        epsilons['contNorm'] = [0.01, 0.001, 0.001]
    else:
        fixedParams['contNorm'] = [1.0, 0.0, 0.0]

    if includeVmic:
        params['Vmic'] = 2e5
        epsilons['Vmic'] = 0.01e5
    else:
        fixedParams['Vmic'] = 2e5
    #Parameters
    if includeMetal:
        params['metal'] = 0.0
        epsilons['metal'] = 0.1
    else:
        fixedParams['metal'] = 0.0

    #multiply epsilons by the coefficient
    for key in epsilons:
        try:
            epsilons[key] *= epsilonsCoefficient
        except TypeError:
            helper = []
            for el in epsilons[key]:
                helper.append(el*epsilonsCoefficient)
            epsilons[key] = helper

    nburn = nchain // 2
    return nchain, nburn, nwalkers, params, epsilons, fixedParams, wlRanges, resultName

##########################################################################
if __name__ == "__main__":

    nchain, nburn, nwalkers, params, epsilons, fixedParams, wlRanges, resultName = prepareParams()

    chain = runMCMC(nchain, nburn, nwalkers, params, epsilons, fixedParams, wlRanges, resultName)
