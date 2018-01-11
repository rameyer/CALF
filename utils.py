import numpy as np
from numpy.random import uniform
from scipy import constants
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.stats import sigmaclip
from scipy.special import wofz
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy.modeling.models import Voigt1D

def cluster_redshifts(redshift_array,delta_z):
	'''
	Given an array of redshifts tracing possible absorptions,
	group them by system based on a redshift tolerance
	Input:
		redshift_array: Explicit.
		delta_z : redshift tolerance
	Return:
		reduced_redshift_array: Explicit.
	'''
	pass

def sample_wavelength_in_velocity(wavelength,velocity):
	'''
	A quick utility to sample a wavelength array in Angstrom at fixed
	velocity intervals. Return indices.
	'''
	indices = []

	dv = 1e-3*constants.c*np.array([(wavelength[i]-wavelength[i-1])/wavelength[i] for i in range(len(wavelength))])
	dv[0] = 0
	v = np.cumsum(dv)
	N = 0
	while N*velocity < np.max(v):
		indices.append(np.min(np.where(v>=N*velocity)))
		N += 1

	return indices


def upsample(array,upsample= 10):
	N = len(array)
	f = interp1d(np.linspace(0,N-1,N), array)
	# Sligthly complicated array to interpolate to keep the original data points in the final answer
	#plt.plot(array,'r')
	#plt.plot(np.linspace(0,N-1,N+(N-1)*(upsample-1)),f(np.linspace(0,N-1,N+(N-1)*(upsample-1))),'b')
	#plt.show()
	return f(np.linspace(0,N-1,N+(N-1)*(upsample-1)))

def doublet_CIV_voigt(wave,logN,b,z,RESVEL,upsampling = 10):
	''' 
	A handy shortcut for the doublet_voigt(...) function containing all the relevant
	physical parameters. See utils::doublet_voigt(...) for more details.
	'''
	Gamma_CIV_1 = 2.642E8
	Gamma_CIV_2 = 2.628E8 
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 
	f_CIV_1548 = 0.1899
	#f_CIV_1550 = 0.09475
	#f_CIV_1548 = 0.194000
	f_CIV_1550 = 0.097000

	return doublet_voigt(wave = wave,logN = logN,b = b,z=z,Gamma_1 = Gamma_CIV_1
		, Gamma_2 = Gamma_CIV_2, lambda_1 =CIV_1,lambda_2 = CIV_2, 
		f_1 = f_CIV_1548,f_2 = f_CIV_1550,RESVEL = RESVEL ,upsampling = upsampling)

def doublet_voigt(wave,logN,b,z,Gamma_1,Gamma_2,lambda_1,lambda_2, f_1,f_2,RESVEL,upsampling = 10):
	'''
	Compute the Voigt profile absorption doublet of a given species, convoluted
	with the given resolution of the intrument.
	First upsamples the wavelength array to avoid any problems arising from 
	the descrete convolution. Then computes the Voigt profile of each transition
	of the doublet, passing in the frequency space and using the real of the 
	Faddeeva function in Voigt(a,x). Performs the instrument resolution 
	convolution and downsamples back to the initial wavelength array length.

	Inputs:
		wave:  N-array with the wavelength in Angstrom
		logN: The integrated strength of the doublet (computed on one 
			  transition only), in cm^{-2}
		b: Doppler parameter of the profile, in km/s
		z: Redshift of the absorption
		Gamma_1 : Lorentz damping of the profile (1st transition)
		Gamma_2 : Lorentz damping of the profile (2nd transition)
		lambda_1: Rest-frame wavelength (Angstrom) of the 1st transition
		lambda_2: Rest-frame wavelength (Angstrom) of the 2nd transition
		f_1 : Oscillator strength of the 1st transition
		f_2 : Oscillator strength of the 2nd transition
		RESVEL: Instrument resolution in km/s
		upsampling: The upsampling factor for the wavelength array 
					necessary to perform a less biaised convolution
	Outputs:
		downsampled: A N-array profile of the convoluted Voigt profile,
					 downsampled back to the input resolution of "wave".
	'''
	#column_conversion_const =  (np.pi*constants.e**2)/(constants.c * constants.m_e) 
	column_conversion_cgs = 2.654e-15

	upsampled_wave = upsample(wave,upsampling)
	nu = constants.c/(upsampled_wave*1e-10)
	dlambda = np.array([upsampled_wave[i]-upsampled_wave[i-1] for i in range(len(upsampled_wave))])
	dlambda[0] = 0

	ratio  =1e-10

	b = b /np.sqrt(2)
	# generic voigt profiles (normalized) 
	V_1 = voigt_faddeeva(x= upsampled_wave-lambda_1*(1+z), y = (1+z)*ratio*lambda_1*b*1e3/constants.c, 
				sigma = (1+z)*lambda_1*b*1e3/constants.c)
	V_2 = voigt_faddeeva(x= upsampled_wave-lambda_2*(1+z), y = (1+z)*ratio*lambda_2*b*1e3/constants.c, 
				sigma = (1+z)*lambda_1*b*1e3/constants.c)

	#tau = (10**logN)*column_conversion_const*(f_1*lambda_1*V_1 + f_2*lambda_2*V_2 )
	voigt_lambda = np.exp(-(10**logN)*column_conversion_cgs*(1e3/constants.c)*(f_1*lambda_1*lambda_1*(1+z)*V_1 + f_2*lambda_2*lambda_2*(1+z)*V_2 )) - 1  
	

	center = (1+z)*(lambda_1+lambda_2)*0.5
	sigma= (1+z)*(lambda_1+lambda_2)*0.5*(RESVEL/2.235)*1e3/constants.c
	
	# Compute gaussian resolution profile
	resolution_profile = gaussian(upsampled_wave, 1/(np.sqrt(2*np.pi)*sigma)  , center,sigma) 
	index_min = np.max(np.where(upsampled_wave < center-20*sigma ))
	index_max =  np.min(np.where(upsampled_wave > center+20*sigma ))
	resolution_profile = resolution_profile[index_min:index_max]    #np.sum( resolution_profile[index_min:index_max])# 

	# Compute dlambda for numerical convolution
	
	convolved_voigt = np.convolve(voigt_lambda*dlambda,resolution_profile, 'same')
	downsampled = convolved_voigt[0:len(convolved_voigt):upsampling]

	#print(np.sum(voigt_lambda*dlambda),np.sum(convolved_voigt*dlambda))

	assert len(wave)==len(downsampled), 'Upsampled+Downsampled array has not the original array lenghth. Check upsampling value.'
	'''
	plt.plot(upsampled_wave,convolved_voigt, 'b')
	plt.plot(upsampled_wave, voigt_lambda,'--k')
	plt.plot(upsampled_wave[index_min:index_max],resolution_profile,'r')
	plt.plot(upsampled_wave,convolved_voigt,'b')
	plt.plot(wave,downsampled,'g')
	plt.show()
	'''
	return downsampled

def voigt_faddeeva(x, y, sigma):
    """
    Real part of Faddeeva function, where    
    w(z) = exp(-z^2) erfc(jz)
    """
    z = (x + 1j*y)/(np.sqrt(2)*sigma)
    return wofz(z).real / (np.sqrt(2*np.pi)*sigma)


def doublet_CIV_gaussian(x,Amp1,x1,sigma):
	f_CIV_1548 = 0.1899
	f_CIV_1550 = 0.09475

	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 

	Amp2 = Amp1 * f_CIV_1550*CIV_2*CIV_2 / (f_CIV_1548 * CIV_1*CIV_1)
	x2 = x1/CIV_1 * CIV_2
	g1 = Amp1*np.exp(-(x-x1)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
	g2 = Amp2*np.exp(-(x-x2)**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)

	return g1+g2

def reduced_chi2_doublet(params,x,flux,error):
	y = doublet_CIV_gaussian(x,params[0],params[1],params[2])
	chi2 = np.sum(np.square(y-flux)/np.square(error))

	return chi2/(len(x)-3.) 

def gaussian(x, Amp,mu, sigma):
	''' A simple gaussian definition with usual parameters'''
	return Amp*np.exp(-(x-mu)**2/(2*sigma**2)) 

def gaussian_fit(x,params):
	return (params[0]*np.exp(-(x-params[1])**2/(2*params[2]**2))) 

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

def gaussian_in_angstrom_normalized_in_v(wave,l0,mu,sigma_kms):
	v = constants.c*1e-3 * l0 / (wave - l0) 
	mu_v = constants.c*1e-3 * l0 / (mu - l0)

	return np.exp(-(v-mu_v)**2/(2*sigma_kms**2)) / (np.sqrt(2*np.pi)*sigma_kms)


def add_CIV(wave,flux,z,strength,b, RESVEL):
	f_CIV_1548 = 0.1899
	f_CIV_1550 = 0.09475

	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 

	#column_conversion_const =  (np.pi*constants.e**2)/(constants.m_e*constants.c*constants.c*1e-3 ) 
	conversion_const_cgs = 2.654e-15

	flux_copy = np.copy(flux)


	# Doppler velocity array (km/s)
	v = constants.c*1e-3 * CIV_1 / (wave - CIV_1)

	# Convolution: b' = b + RESVEL [km/s]
	width_kms = np.sqrt((0.5*b)**2  + RESVEL**2)
	width_kms_z = width_kms * (1+z)
	# To angstrom
	width_angstrom = (1+z)*CIV_1*width_kms*1e3/constants.c
	print(width_angstrom)

	tau_1 = gaussian_in_angstrom_normalized_in_v(wave,l0=CIV_1,mu=(1+z)*CIV_1,sigma_kms= width_kms_z)
	tau_2 = gaussian_in_angstrom_normalized_in_v(wave,l0=CIV_2,mu=(1+z)*CIV_2,sigma_kms= width_kms_z)

	amplitude_1 = (10**strength) * CIV_1 * f_CIV_1548 * conversion_const_cgs
	amplitude_2 = (10**strength) * CIV_2 * f_CIV_1550 * conversion_const_cgs

	plt.plot(v,tau_1+tau_2)
	plt.show()

	I =amplitude_1* np.exp(-tau_1) + amplitude_2*np.exp(-tau_2)
	

	plt.plot(wave,I) 
	plt.show()

	#flux_copy = flux_copy + tau
	return I

def logN_to_amp(logN,b, wavelengths):
	###### Constants ######
	f_CIV_1548 = 0.1899
	f_CIV_1550 = 0.09475
	column_conversion_const =  (np.pi*constants.e**2)/(constants.c * constants.m_e) 
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 
	###### Conversion #####

	tau = 10**(logN)*(f_CIV_1548 * CIV_1 * column_conversion_const)


	return -(1-np.exp(-tau))

def fit_sigma_clip(wave, flux, error, x0, bounds,  method = 'TNC' ,
				   sigma = 2, iters = 3,fun = gaussian):
		
		temp_wave = wave
		temp_flux = flux
		temp_error = error
		init = x0

		for i in range(iters):

			popt, pcov = curve_fit(f = fun, xdata = temp_wave,
									ydata = temp_flux, p0 = init, sigma= temp_error,
									absolute_sigma = True, bounds = bounds, 
									method = method)

			fit_flux = gaussian(temp_wave,*popt)

			std = np.std(fit_flux-temp_flux)
			new_indices = [k for k in range(len(temp_wave)) 
							if np.abs(temp_flux[k]-fit_flux[k]) < sigma*std]
			if new_indices == []:
				return popt,pcov,temp_wave,temp_flux,temp_error
			else:
				old_indices = np.array(new_indices)
				#print (i,new_indices)
				temp_wave = temp_wave[old_indices]
				temp_flux = temp_flux[old_indices]
				temp_error = temp_error[old_indices]
				init = popt
		return popt,pcov,temp_wave,temp_flux,temp_error

def log_inverse_gaussian(temp_wave,params):
	return np.log(1/ (1+(params[0]*np.exp(-(temp_wave-params[1])**2/(2*params[2]**2)))) )

def log_column_density_from_gaussian(temp_wave,popt,perr,redshift,wavelength,transition_strength):

	column_conversion_cgs = 2.654e-15

	density  = quad(log_inverse_gaussian,np.min(temp_wave),np.max(temp_wave),args=popt)[0] / (column_conversion_cgs*
						(1e3/constants.c)*transition_strength*wavelength*wavelength*(1+redshift) ) 
	popt_plus = popt
	popt_plus[0] = popt_plus[0] + 2*perr[0] 
	popt_plus[2] = popt_plus[2] + 2*perr[2] 
	plus_1_sigma_density = quad(log_inverse_gaussian,np.min(temp_wave),np.max(temp_wave),args=popt_plus)[0] / (column_conversion_cgs*
						(1e3/constants.c)*transition_strength*wavelength*wavelength*(1+redshift) ) 
	popt_minus = popt
	popt_minus[0] = popt_minus[0] - 2*perr[0] 
	popt_minus[2] = popt_minus[2] - 2*perr[2] 
	minus_1_sigma_density = quad(log_inverse_gaussian,np.min(temp_wave),np.max(temp_wave),args=popt_minus)[0] / (column_conversion_cgs*
						(1e3/constants.c)*transition_strength*wavelength*wavelength*(1+redshift) ) 


	return np.log10(density),np.log10(plus_1_sigma_density),np.log10(minus_1_sigma_density)


def logN(wave,flux,err,delta_width, RESVEL,window_size = 40, column_error_max=0.1 ):
	###### Constants ######
	f_CIV_1548 = 0.1899
	f_CIV_1550 = 0.09475
	column_conversion_const =  (np.pi*constants.e**2)/(constants.c * constants.m_e) 
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 
	# FWHM to sigma
	s_RESVEL = RESVEL / 2.235

	###### Conversion #####
	logN = np.zeros(len(wave))
	#dvperpix = 1e-3*constants.c*np.array([(wave[i]-wave[i-1])/wave[i] for i in range(len(wave))])
	#dvperpix[0] = dvperpix[1]
	
	#step_pixel = int(np.max((np.ceil(delta_width*1e3*np.max(wave)/constants.c),1)))
	fitting_locations_indices = sample_wavelength_in_velocity(wave,delta_width)
	fitting_locations_indices = np.array([i for i in fitting_locations_indices if 
										(i > window_size and i< len(wave)-window_size) ])
	#range(window_size,len(wave)-window_size,step_pixel)
	#print(len(fitting_locations_indices),len(wave))

	logN_1 = np.zeros(len(wave))
	logN_2 = np.zeros(len(wave))
	logN_1_err = np.zeros(len(wave))
	logN_2_err = np.zeros(len(wave))
	wave_c = np.zeros(len(wave))

	for i in fitting_locations_indices:
		temp_wave = wave[int(i-window_size*0.5):int(i+window_size*0.5)]
		temp_flux = flux[int(i-window_size*0.5):int(i+window_size*0.5)] -1
		temp_err = err[int(i-window_size*0.5):int(i+window_size*0.5)]
		z_min = wave[i]/CIV_1 -1
		z_max = wave[i]/CIV_2 -1
		popt,pcov,_,_,_ = fit_sigma_clip(temp_wave,temp_flux, temp_err, 
			   iters = 5, sigma = 3, fun = gaussian, x0 = [-0.2,wave[i],np.sqrt(20.0**2+s_RESVEL**2)*1e3*wave[i]/constants.c], 
			   method = 'trf' , bounds = ( [-1,wave[i]*(1-5*1e3/constants.c), np.sqrt(5**2+s_RESVEL**2)*1e3*wave[i]/constants.c],
			   	[0, wave[i]*(1+5.0*1e3/constants.c), np.sqrt(40.0**2+s_RESVEL**2)*1e3*wave[i]/constants.c] )  )

		perr = np.sqrt(np.diag(pcov))

		if True:

			log_density_1,log_density_1_plus,log_density_1_minus = log_column_density_from_gaussian(temp_wave=temp_wave,popt=popt,
												perr=perr,redshift=z_min,wavelength=CIV_1,transition_strength=f_CIV_1548)
			
			logN_1[i] = log_density_1
			logN_1_err[i] = 0.5*(np.abs(log_density_1_plus-log_density_1) + np.abs(log_density_1_minus-log_density_1) ) 
			
			log_density_2,log_density_2_plus,log_density_2_minus = log_column_density_from_gaussian(temp_wave=temp_wave,popt=popt,
												perr=perr,redshift=z_max,wavelength=CIV_2,transition_strength=f_CIV_1550)
			logN_2[i] = log_density_2
			logN_2_err[i] = 0.5*(np.abs(log_density_2_plus-log_density_2) + np.abs(log_density_2_minus-log_density_2) ) 
			

	temp_log = np.array([[lN1,lN2,err1,err2,w] for lN1,lN2,err1,err2,w 
				in zip(logN_1,logN_2,logN_1_err, logN_2_err,wave) 
				if lN1 >0 and lN2 >0 and err1 <= column_error_max and err2<=column_error_max ])
	#print (temp_log)
	if len(temp_log) == 1:
		return temp_log[0], temp_log[1], temp_log[2], temp_log[3], temp_log[4]
	else:
		return temp_log[:,0], temp_log[:,1], temp_log[:,2], temp_log[:,3], temp_log[:,4]

def potential_CIV_from_column_densities(wave_c,columns_1,columns_2,error_1,error_2,
								RESVEL, threshold_columns = 13, tol_columns = 0.05):
	# Constants
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740

	redshifts = []

	thresholded_columns_1 = np.array([[c1,err,w]  for c1,err,w in zip(columns_1,error_1,wave_c) if c1 > threshold_columns-err])
	for tc1 in thresholded_columns_1:
		potential_columns = np.array([c2 for c2,err2,w2 in zip(columns_2,error_2,wave_c) 
				if (np.abs(tc1[2]/CIV_1-w2/CIV_2) <= 2*(w2/CIV_1)*(RESVEL*1e3/constants.c) and
					np.abs(tc1[0] - c2) < tc1[1]+err2+tol_columns ) ])
		if len(potential_columns)>0:
			redshifts.append(tc1[2]/CIV_1 - 1)
	return redshifts

def guess_CIV(wave,flux,err, column_threshold = 13,
				column_error_max = 0.2,RESVEL = 30.0, window_size = 20):
	redshift_CIV_guesses = []
	# Constants
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740
	# Columns_1
	columns_1, columns_2,error_1,error_2, wave_c = logN(wave,flux,err,delta_width = 10,RESVEL = RESVEL,
														column_error_max = column_error_max, window_size=window_size)
	# Find paired densities + thresholding
	CIV_redshifts = potential_CIV_from_column_densities(wave_c,columns_1,
						columns_2, error_1,error_2, threshold_columns = column_threshold,
						tol_columns = 0.0, RESVEL = RESVEL)

	#print(CIV_redshifts)
	plt.errorbar(wave_c, columns_1,error_1)
	plt.errorbar(wave_c, columns_2,error_2)
	plt.plot(wave,flux)
	plt.scatter((1+np.array(CIV_redshifts))*CIV_1, np.ones(len(CIV_redshifts))*1.5)
	plt.show()
	return CIV_redshifts	
	


	



	