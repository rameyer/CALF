import numpy as np
from astropy.modeling.functional_models import Voigt1D

def voigt_approx(x,Amp, x0, fwhm_L, fwhm_G):
	v = Voigt1D(x0,Amp,fwhm_L,fwhm_G)
	return v(x)


def doublet_voigt_approx(x,Amp1,x1,fwhm_L,fwhm_G):
	'''
	Since the width/location of doublet is fixed, the profile is 
	parametrized only by the first feature parameters.
	'''
	f_CIV_1548 = 0.194000
	f_CIV_1550 = 0.097000

	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 

	Amp2 = Amp * f_CIV_1550*CIV_2 / (f_CIV_1548 * CIV_1)
	x2 = x1/CIV_1 * CIV_2

	v1 = voigt_approx(x1,Amp1,fwhm_L,fwhm_G)
	v2 = voigt_approx(x2,Amp2,fwhm_L,fwhm_G)

	return v1(x) + v2(x)

#def CIV_doublet(lamba1,amp1,z,sigma, ratio):

def gaussian(x, Amp,mu, sigma, Const):
	''' A simple gaussian definition with usual parameters'''
	return Amp*np.exp(-(x-mu)**2/sigma**2) + Const

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
	pairs_1 = []
	pairs_2 = []
	for a1 in array1:
		temp_pairs = np.array([[a1,a2] for a2 in array2 if np.abs(a1-a2) < threshold])
		if temp_pairs != []:
			pairs_1 = np.concatenate([pairs_1,temp_pairs[:,0]])
			pairs_2 = np.concatenate([pairs_2,temp_pairs[:,1]])

	pairs = np.zeros([len(pairs_1),2])
	pairs[:,0] = pairs_1
	pairs[:,1] = pairs_2

	return pairs

def find_potential_doublets(wavelengths, lambda1,lambda2, z_threshold):
	'''
	From a number of wavelengths, find pairs of possible doublets given the 
	doublet wavelength and a redshift threshold cut.
	Return an Nx2 array with indices of the pairs in the original array.
	'''

	_ , unique_indices = np.unique(np.around(gaussian_params[:,1],1), return_index = True)

	pairs_1 = []
	pairs_2 = []
	for i1 in unique_indices:
		temp_pairs = np.array([[i1,i2] for i2 in unique_indices 
			if np.abs(gaussian_params[i1,1]/lambda1-gaussian_params[i2,1]/lambda2) < z_threshold])
		if temp_pairs != []:
			pairs_1 = np.concatenate([pairs_1,temp_pairs[:,0]])
			pairs_2 = np.concatenate([pairs_2,temp_pairs[:,1]])

	pairs = np.zeros([len(pairs_1),2])
	pairs[:,0] = pairs_1
	pairs[:,1] = pairs_2

	return pairs.astype(int)

	#return np.array([[np.where(wavelengths/lambda1-1==l[0]),np.where(wavelengths/lambda2-1==l[1])] for l in redshift_pairs])

def running_median(array,width):
	median = np.zeros(len(array))

	median[width:len(array)-width] = np.array([np.median(array[j-width:j+width]) 
		for j in range(width, len(array)-width)])

	return median




