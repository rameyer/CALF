import numpy as np
from numpy.random import uniform
from scipy import constants
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.stats import sigmaclip

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

def gaussian(x, Amp,mu, sigma):
	''' A simple gaussian definition with usual parameters'''
	return Amp*np.exp(-(x-mu)**2/(2*sigma**2)) 

def reduced_chi2_gaussian(params,x,flux,error):
	y =  params[0]*np.exp(-(x-params[1])**2/(2*params[2]**2)) 

	chi2 = np.sum(np.square(y-flux)/np.square(error))

	return chi2/(len(x)-3)

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

def fit_sigma_clip(wave, flux, error,
				   sigma = 2, iters = 3,fun = reduced_chi2_doublet, x0 = [-0.2,1e4,0.1], 
				   method = 'TNC' , bounds = ( (-1,0), (1e4-2,1e4+2),(5,60))  ):
		
		temp_wave = wave
		temp_flux = flux
		temp_error = error
		init = x0

		for i in range(iters):
			res = minimize(fun = reduced_chi2_doublet, x0 = init, 
					   args = (temp_wave,temp_flux, temp_error),
					   method = method , bounds = bounds)
			fit_flux = doublet_CIV_gaussian(temp_wave,*res.x)
			#plt.plot(temp_wave,fit_flux)
			#plt.plot(temp_wave,temp_flux)
			#plt.show()

			std = np.std(fit_flux-temp_flux)
			new_indices = [k for k in range(len(temp_wave)) if np.abs(temp_flux[k]-fit_flux[k]) < sigma*std]
			if new_indices == []:
				continue
			else:
				new_indices = np.array(new_indices)
				#print (i,new_indices)
				temp_wave = temp_wave[new_indices]
				temp_flux = temp_flux[new_indices]
				temp_error = temp_error[new_indices]
				init = res.x
		return res,temp_wave,temp_flux,temp_error

def logN(wave,flux,err,delta_width, window_size = 40):
	###### Constants ######
	f_CIV_1548 = 0.194000
	f_CIV_1550 = 0.097000
	column_conversion_const =  (np.pi*constants.e**2)/(constants.c * constants.m_e) 
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 
	###### Conversion #####
	logN = np.zeros(len(wave))
	dvperpix = 1e-3*constants.c*np.array([(wave[i]-wave[i-1])/wave[i] for i in range(len(wave))])
	dvperpix[0] = dvperpix[1]
	RESVEL = np.mean(dvperpix)
	step_pixel = int(np.max((np.ceil(delta_width/np.mean(dvperpix)),1)))
	#wave_bin = wave[range(window_size,len(wave)-window_size,step_pixel)]
	logN_1 = np.zeros(len(wave))
	logN_2 = np.zeros(len(wave))
	
	for i in range(window_size,len(wave)-window_size,step_pixel):
		temp_wave = wave[int(i-window_size*0.5):int(i+window_size*0.5)]
		temp_flux = flux[int(i-window_size*0.5):int(i+window_size*0.5)] -1
		temp_err = err[int(i-window_size*0.5):int(i+window_size*0.5)]
		res,_,_,_ = fit_sigma_clip(temp_wave,temp_flux, temp_err, 
			   iters = 5, sigma = 2,fun = reduced_chi2_gaussian, x0 = [-0.2,wave[i],20.0/RESVEL], 
			   method = 'TNC' , bounds = ( (-1,0),(wave[i]-1e-15,wave[i]), (5/RESVEL,40/RESVEL))  )

		tau = -np.log(1+res.x[0])
		if (1+res.x[0]) > 3*err[i]:
			integral_tau = -res.x[0]*np.sqrt(2*np.pi)*res.x[2]
			#print tau, int_tau
			logN_1[i] = -np.log10((f_CIV_1548 * CIV_1 * column_conversion_const) /integral_tau)
			logN_2[i] = -np.log10((f_CIV_1550 * CIV_2 * column_conversion_const) /integral_tau)


	wave_c = np.array([w for w,c in zip(wave,logN_1) if c != 0])
	logN_1 = np.array([lN for lN in logN_1 if lN != 0.0])
	logN_2 = np.array([lN for lN in logN_2 if lN != 0.0])
	return logN_1,logN_2,wave_c

def potential_CIV_from_column_densities(wave_c,columns_1,columns_2,threshold_columns = 12.5,
								 tol_columns = 0.05, tol_redshift = 0.001):
	# Constants
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740

	redshifts = []

	#local_max_indices = argrelmax(columns_1*(columns_1>threshold_columns), order = 5)
	#thresholded_columns_1 = np.array(zip(columns_1[local_max_indices],wave_c[local_max_indices]))
	thresholded_columns_1 = np.array([[c1,w]  for c1,w in zip(columns_1,wave_c) if c1 > threshold_columns])
	for tc1 in thresholded_columns_1:
		potential_columns = np.array([c2 for c2,w2 in zip(columns_2,wave_c) 
				if (np.abs(tc1[1]/CIV_1-w2/CIV_2) < tol_redshift and
					np.abs(tc1[0] - c2) < tol_columns) ])
		if len(potential_columns)>0:
			redshifts.append(tc1[1]/CIV_1 - 1)
	return redshifts

def guess_CIV(wave,flux,err, SNR_threshold = 3, z_threshold = 0.0005,
				chi2_threshold = 6,RESVEL = 30.0, window_size = 20):
	redshift_CIV_guesses = []
	# Constants
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740
	# Columns_1
	columns_1, columns_2,wave_c = logN(wave,flux,err,delta_width = 20,window_size=window_size)
	# Find paired densities + thresholding
	CIV_redshifts = potential_CIV_from_column_densities(wave_c,columns_1,columns_2,
						threshold_columns = 13,tol_columns = 0.1, tol_redshift = 0.001)
	print CIV_redshifts

	#plt.plot(wave_c, columns_1)
	#plt.plot(wave_c, columns_2)
	#plt.plot(wave,flux)
	#plt.scatter((1+np.array(CIV_redshifts))*CIV_1, np.ones(len(CIV_redshifts))*1.5)
	#plt.show()
	return CIV_redshifts	
	'''
	# Run matched filter
	kernel = gaussian( np.linspace(0,20,20),-0.2, 10,20/RESVEL,0)
	SNR = matched_filter(kernel,flux-1,err)
	#plt.plot(wave,flux)
	#plt.plot(wave,SNR)
	#plt.show()
	peaks_indices = argrelmax(*(SNR>SNR_threshold), order = 5)
	peaks = wave[peaks_indices]
	#print peaks
	# Call find_potential_doublets
	pairs_indices = find_potential_doublets(wave[peaks_indices],CIV_1,CIV_2,z_threshold)
	print peaks[pairs_indices[:,0]]
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
		res,temp_wave,temp_flux,temp_error = fit_sigma_clip(temp_wave,temp_flux, temp_error, 
					   iters = 3, sigma = 2,fun = reduced_chi2_doublet, x0 = [-0.2,peak,20.0/RESVEL], 
					   method = 'TNC' , bounds = ( (-1,0), (peak-2,peak+2),
					   (5/RESVEL,60/RESVEL))  )
		#res = minimize(fun = reduced_chi2_doublet, x0 = [-0.2,peak,20.0/RESVEL], 
		#			   args = (temp_wave,temp_flux, temp_error),
		#			   method = 'TNC' , bounds = ( (-1,0), (peak-2,peak+2),
		#			   (5/RESVEL,60/RESVEL))  )
		#print peak/CIV_1-1,reduced_chi2_doublet(res.x,temp_wave,temp_flux, temp_error )
		if reduced_chi2_doublet(res.x,temp_wave,temp_flux, temp_error ) < chi2_threshold:
			redshift_CIV_guesses.append(res.x[1]/CIV_1 -1)
			#print(res.x[1]/CIV_1 -1)
			#plt.plot(temp_wave,temp_flux)
			#plt.plot(temp_wave,doublet_CIV_gaussian(temp_wave, *res.x))
			#plt.show()

	# Return the redshift of the selected doublets
	return redshift_CIV_guesses
	'''

def plot_completeness():
	# Create array to store completeness
	# Loop over b,N
	# find if redshift is matching
	# count
	inputs = np.loadtxt('./Completeness/input_logN_13.1_b_20.txt')
	guesses = np.loadtxt('./Completeness/guesses_logN_13.1_b_20.txt')
	print(len(guesses), len(inputs))



	



	