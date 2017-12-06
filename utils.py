import numpy as np
from numpy.random import uniform
from scipy import constants
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

def doublet_CIV_gaussian(x,Amp1,x1,b):
	f_CIV_1548 = 0.194000
	f_CIV_1550 = 0.097000

	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 

	Amp2 = Amp1 * f_CIV_1550*CIV_2 / (f_CIV_1548 * CIV_1)
	x2 = x1/CIV_1 * CIV_2
	g1 = Amp1*np.exp(-(x-x1)**2/(2*b**2))
	g2 = Amp2*np.exp(-(x-x2)**2/(2*b**2))
	
	return g1+g2

def reduced_chi2_doublet(params,x,flux,error):
	y = doublet_CIV_gaussian(x,params[0],params[1],params[2])
	chi2 = np.sum(np.square(y-flux)/np.square(error))

	return chi2/(len(x)-3.) 

def gaussian(x, Amp,mu, sigma, Const):
	''' A simple gaussian definition with usual parameters'''
	return Amp*np.exp(-(x-mu)**2/(2*sigma**2)) + Const

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

def CIV_matched_filter(wave, flux, err,amp,b):
	''' Matched Filtering: e.g. Hewett 1985, Bolton 2004. 
	Inputs:
	u: kernel
	f: flux of the spectra
	e: error
	Returns:
	SNR: Signal to noise array for all flux values
	'''
	width = 40

	Cj1 = np.array([np.sum(doublet_CIV_gaussian(wave[j-width:j+width],amp,wave[j],b)*flux[j-width:j+width]/err[j-width:j+width]**2)
		for j in range(len(wave))])
	Cj2 = np.array([np.sum(doublet_CIV_gaussian(wave[j-width:j+width],amp,wave[j],b)**2/err[j-width:j+width]**2)
		for  j in range(len(wave))])
	SNR = Cj1 / np.sqrt(Cj2)

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

	_ , unique_indices = np.unique(np.around(wavelengths), return_index = True)

	pairs_1 = []
	pairs_2 = []
	for i1 in unique_indices:
		temp_pairs = np.array([[i1,i2] for i2 in unique_indices 
			if np.abs(wavelengths[i1]/lambda1-wavelengths[i2]/lambda2) < z_threshold])
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

def add_CIV(wave,flux,z,amp,b):
	CIV_1 = 1548.2020    

	flux_copy = np.copy(flux)

	flux_copy = flux_copy + doublet_CIV_gaussian(wave,amp,CIV_1*(1+z),b)

	return flux_copy

def logN_to_amp(logN,b, wavelengths):
	###### Constants ######
	f_CIV_1548 = 0.194000
	f_CIV_1550 = 0.097000
	column_conversion_const =  (np.pi*constants.e**2)/(constants.c * constants.m_e) 
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 
	###### Conversion #####

	tau = 10**(logN)*(f_CIV_1548 * CIV_1 * column_conversion_const)


	return -(1-np.exp(-tau))

def guess_CIV(wave,flux,err, SNR_threshold = 3, z_threshold = 0.0005,
			  chi2_threshold = 6,RESVEL = 30.0, window_size = 40):
	redshift_CIV_guesses = []
	# Constants
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 

	# Run matched filter
	kernel = gaussian( np.linspace(0,20,20),-0.2, 10,20/RESVEL,0)
	SNR = matched_filter(kernel,flux-1,err)
	#plt.plot(wave,flux)
	#plt.plot(wave,SNR)
	#plt.show()
	peaks_indices = argrelmax(SNR*(SNR>SNR_threshold), order = 5)
	peaks = wave[peaks_indices]
	# Call find_potential_doublets
	pairs_indices = find_potential_doublets(wave[peaks_indices],CIV_1,CIV_2,z_threshold)
	# Fit the voigt profile and cut in chi^2

	for i in range(len(pairs_indices)):
		peak = peaks[pairs_indices[i,0]	]
		index_peak1 = np.where(wave == peak)[0][0]
		index_peak2 = np.where(wave == peaks[pairs_indices[i,1]	])[0][0]
		if SNR[index_peak1] <= SNR[index_peak2]:
			#print 'eliminated pair at z' , peak/CIV_1-1
			continue

		temp_wave = wave[int(index_peak1-window_size*0.5):int(index_peak2+window_size*0.5)]
		temp_flux = flux[int(index_peak1-window_size*0.5):int(index_peak2+window_size*0.5)] - 1
		temp_error = err[int(index_peak1-window_size*0.5):int(index_peak2+window_size*0.5)]
		res = minimize(fun = reduced_chi2_doublet, x0 = [-0.2,peak,10.0/RESVEL], 
					   args = (temp_wave,temp_flux, temp_error),
					   method = 'TNC' , bounds = ( (-1,0), (peak-2,peak+2),
					   (10/RESVEL,60/RESVEL))  )
		#print peak/CIV_1-1,reduced_chi2_doublet(res.x,temp_wave,temp_flux, temp_error )
		if reduced_chi2_doublet(res.x,temp_wave,temp_flux, temp_error ) < chi2_threshold:
			redshift_CIV_guesses.append(res.x[1]/CIV_1 -1)
			#print res.x[1]/CIV_1 -1
			#plt.plot(temp_wave,temp_flux)
			#plt.plot(temp_wave,doublet_CIV_gaussian(temp_wave, *res.x))
			#plt.show()

	# Return the redshift of the selected doublets
	return redshift_CIV_guesses


def plot_completeness():
	# Create array to store completeness
	# Loop over b,N
	# find if redshift is matching
	# count
	inputs = np.loadtxt('./Completeness/input_logN_13.1_b_20.txt')
	guesses = np.loadtxt('./Completeness/guesses_logN_13.1_b_20.txt')
	print len(guesses), len(inputs)



	



	

