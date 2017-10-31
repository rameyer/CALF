import numpy as np

#def CIV_doublet(lamba1,amp1,z,sigma, ratio):

def gaussian(Amp,mu, sigma,x):
	''' A simple gaussian definition with usual parameters'''
	return Amp*np.exp(-(x-mu)**2/sigma**2)

def matched_filter(u, f, e):
	''' Matched Filtering: e.g. Hewett 1985, Bolton 2004. 
	Inputs:
	u: kernel
	f: flux of the spectra
	e: error
	Returns:
	SNR: Signal to noise array for all flux values
	'''
	width = int(len(u)/2)

	Cj1 = np.array([np.sum(u*f[j-width:j+width]/e[j-width:j+width]**2)
		for j in range(width,int(len(f) - width))])
	Cj2 = np.array([np.sum(u**2/e[j-width:j+width]**2)
		for j in  range(width,int(len(f) - width))])

	SNR = np.zeros(len(f))
	SNR[width: len(f)-width] = Cj1 / np.sqrt(Cj2)

	return SNR

def find_pairs(array1, array2, threshold):
	'''
	Find all pairs of values a1, a2 such that a1-a2 > threshold 
	'''
	pairs = []
	for a1 in array1:
		temp_pairs_a1 = [a1 for a2 in array2 if a1-a2 > threshold]
		if temp_pairs_a1 != []:
			pairs = np.concatenate([pairs,temp_pairs_a1])

	return pairs

def find_potential_doublets(wavelengths, lambda1,lambda2, z_threshold):
	'''
	From a number of wavelengths, find pairs of possible doublets given the 
	doublet wavelength and a redshift threshold cut.
	'''
	return find_pairs(wavelengths/lambda1-1,wavelengths/lambda2-1,z_threshold)

def running_median(array,width):
	median = np.zeros(len(array))

	median[width:len(array)-width] = np.array([np.median(array[j-width:j+width]) 
		for j in range(width, len(array)-width)])

	return median




