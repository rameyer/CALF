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
	delta_z 
	return 0


def upsample(array,upsample= 10):
	N = len(array)
	f = interp1d(np.linspace(0,N-1,N), array)
	# Sligthly complicated array to interpolate to keep the original data points in the final answer
	return f(np.linspace(0,N-1,N+(N-1)*(upsample-1)))


def doublet_CIV_voigt(wave,logN,b,z,RESVEL,upsampled = 10):
	Gamma_CIV_1 = 2.642E8
	Gamma_CIV_2 = 2.628E8 
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 
	f_CIV_1548 = 0.1899
	f_CIV_1550 = 0.09475
	sigma0 = 0.0263 #cm/s 

	upsampled_wave = upsample(wave,upsampled)
	nu = constants.c/(upsampled_wave*1e-10)

	# converting b from km/s to cm/s
	V_1 = Voigt(b = b*1e5, z=z, nu = nu, nu0 = constants.c/(CIV_1*1e-10),gamma= Gamma_CIV_1)
	V_2 = Voigt(b = b*1e5, z=z, nu = nu, nu0 = constants.c/(CIV_2*1e-10),gamma= Gamma_CIV_2)

	tau = (10**logN)*sigma0*(f_CIV_1548*V_1 +f_CIV_1550*V_2 )
	voigt_lambda = np.exp(-tau) - 1

	
	center = (1+z)*CIV_1
	sigma= (1+z)*CIV_1*RESVEL*1e3/constants.c
	
	resolution_profile = gaussian(upsampled_wave, 1, center,sigma) 
	index_min = np.max(np.where(upsampled_wave < center-3*sigma ))
	index_max =  np.min(np.where(upsampled_wave > center+3*sigma ))
	resolution_profile = resolution_profile[index_min:index_max] / np.sum( resolution_profile[index_min:index_max])

	convolved_voigt = np.convolve(voigt_lambda,resolution_profile, 'same')
	downsampled = convolved_voigt[0:len(convolved_voigt):upsampled]

	assert len(wave)==len(downsampled), 'Upsampled+Downsampled array has not the original array lenghth. Check upsampling value.'

	#plt.plot(upsampled_wave,voigt_lambda,'--k')
	#plt.plot(upsampled_wave[index_min:index_max],resolution_profile,'r')
	#plt.plot(upsampled_wave,convolved_voigt,'b')
	#plt.plot(wave,downsampled,'g')
	#plt.show()

	return downsampled

def voigt(x, a):
    """
    Real part of Faddeeva function, where    
    w(z) = exp(-z^2) erfc(jz)
    """
    z = x + 1j*a
    return wofz(z).real

def Voigt(b, z, nu, nu0, gamma):
    """
    Generate Voigt Profile for a given transition
    Parameters:
    ----------
    b: float
        Doppler parameter of the voigt profile
    z: float
        resfhit of the absorption line
    nu: array_like
        rest frame frequncy array
    nu0: float
        rest frame frequency of transition [1/s]
    gamma: float
        Damping coefficient (transition specific)
    Returns:
    ----------
    V: array_like
        voigt profile as a function of frequency
    """

    delta_nu = nu - nu0 / (1+z)
    delta_nuD = b * nu / (constants.c*1e2 )
    
    prefactor = 1.0 / ((np.pi**0.5)*delta_nuD)
    x = delta_nu/delta_nuD
    a = gamma/(4*np.pi*delta_nuD)

    return prefactor * voigt(x,a) 



def doublet_CIV_gaussian(x,Amp1,x1,sigma):
	f_CIV_1548 = 0.194000
	f_CIV_1550 = 0.09700

	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 

	Amp2 = Amp1 * f_CIV_1550*CIV_2 / (f_CIV_1548 * CIV_1)
	x2 = x1/CIV_1 * CIV_2
	g1 = Amp1*np.exp(-(x-x1)**2/(2*sigma**2))
	g2 = Amp2*np.exp(-(x-x2)**2/(2*sigma**2))
	
	return g1+g2

def reduced_chi2_doublet(params,x,flux,error):
	y = doublet_CIV_gaussian(x,params[0],params[1],params[2])
	chi2 = np.sum(np.square(y-flux)/np.square(error))

	return chi2/(len(x)-3.) 

def gaussian(x, Amp,mu, sigma):
	''' A simple gaussian definition with usual parameters'''
	return (Amp)*np.exp(-(x-mu)**2/(2*sigma**2)) 

#def gaussian_fit(x,params):
#	return (params[0]/np.sqrt(2*np.pi*params[2]**2))*np.exp(-(x-params[1])**2/(2*params[2]**2)) 

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

def add_CIV(wave,flux,z,strength,b, RESVEL, dvperpix):
	CIV_1 = 1548.2020    

	flux_copy = np.copy(flux)

	# Convolution: b' = b + RESVEL [km/s]
	width_convolved = np.sqrt(b**2 + RESVEL**2)
	amplitude = logN_to_amp(strength,width_convolved,(1+z)*CIV_1)
	# Bin it, divide by dv per pixel
	width_pixel = width_convolved/dvperpix

	flux_copy = flux_copy + doublet_CIV_gaussian(wave,amplitude,CIV_1*(1+z),width_pixel)

	return flux_copy

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
			'''
			res = minimize(fun = reduced_chi2_doublet, x0 = init, 
					   args = (temp_wave,temp_flux, temp_error),
					   method = method , bounds = bounds)
			'''
			fit_flux = doublet_CIV_gaussian(temp_wave,*popt)
			#plt.plot(temp_wave,fit_flux)
			#plt.plot(temp_wave,temp_flux)
			#plt.show()

			std = np.std(fit_flux-temp_flux)
			new_indices = [k for k in range(len(temp_wave)) if np.abs(temp_flux[k]-fit_flux[k]) < sigma*std]
			if new_indices == []:
				continue
			else:
				old_indices = np.array(new_indices)
				#print (i,new_indices)
				temp_wave = temp_wave[old_indices]
				temp_flux = temp_flux[old_indices]
				temp_error = temp_error[old_indices]
				init = popt
		return popt,pcov,temp_wave,temp_flux,temp_error

def logN(wave,flux,err,delta_width, RESVEL,window_size = 40, column_error_max=0.1 ):
	###### Constants ######
	f_CIV_1548 = 0.1899
	f_CIV_1550 = 0.09475
	column_conversion_const =  (np.pi*constants.e**2)/(constants.c * constants.m_e) 
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740 
	###### Conversion #####
	logN = np.zeros(len(wave))
	dvperpix = 1e-3*constants.c*np.array([(wave[i]-wave[i-1])/wave[i] for i in range(len(wave))])
	dvperpix[0] = dvperpix[1]
	step_pixel = int(np.max((np.ceil(delta_width/np.mean(dvperpix)),1)))
	
	logN_1 = np.zeros(len(wave))
	logN_2 = np.zeros(len(wave))
	logN_1_err = np.zeros(len(wave))
	logN_2_err = np.zeros(len(wave))
	
	for i in range(window_size,len(wave)-window_size,step_pixel):
		temp_wave = wave[int(i-window_size*0.5):int(i+window_size*0.5)]
		temp_flux = flux[int(i-window_size*0.5):int(i+window_size*0.5)] -1
		temp_err = err[int(i-window_size*0.5):int(i+window_size*0.5)]
		popt,pcov,_,_,_ = fit_sigma_clip(temp_wave,temp_flux, temp_err, 
			   iters = 5, sigma = 2, fun = gaussian, x0 = [-0.2,wave[i],(20.0+RESVEL)/dvperpix[i]], 
			   method = 'trf' , bounds = ( [-1,wave[i]-1e-10, (5.0+RESVEL)/dvperpix[i]],
			   	[0, wave[i], (40.0+RESVEL)/dvperpix[i]] )  )
		perr = np.sqrt(np.diag(pcov))

		tau = -np.log(1+popt[0])
		if (1+popt[0]) > 3*err[i]:
			integral_tau = -popt[0]*np.sqrt(2*np.pi)*popt[2]
			#print tau, int_tau
			logN_1[i] = -np.log10((f_CIV_1548 * CIV_1 * column_conversion_const) /integral_tau)
			logN_2[i] = -np.log10((f_CIV_1550 * CIV_2 * column_conversion_const) /integral_tau)
			logN_1_err[i] = np.abs(perr[0]/popt[0] + perr[2]/popt[2])
			logN_2_err[i] = np.abs(perr[0]/popt[0] + perr[2]/popt[2])


	#wave_c = np.array([w for w,c in zip(wave,logN_1) if c != 0])
	temp_log = np.array([[lN1,lN2,err1,err2,w] for lN1,lN2,err1,err2,w 
				in zip(logN_1,logN_2,logN_1_err, logN_2_err,wave) 
				if lN1 >10 and lN2 >10 and err1 <= column_error_max and err2<=column_error_max ])
	
	return temp_log[:,0], temp_log[:,1], temp_log[:,2], temp_log[:,3], temp_log[:,4]

def potential_CIV_from_column_densities(wave_c,columns_1,columns_2,error_1,error_2,
								threshold_columns = 13, tol_columns = 0.05, tol_redshift = 0.0005):
	# Constants
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740

	redshifts = []

	thresholded_columns_1 = np.array([[c1,err,w]  for c1,err,w in zip(columns_1,error_1,wave_c) if c1 > threshold_columns-err])
	for tc1 in thresholded_columns_1:
		potential_columns = np.array([c2 for c2,err2,w2 in zip(columns_2,error_2,wave_c) 
				if (np.abs(tc1[2]/CIV_1-w2/CIV_2) < tol_redshift and
					np.abs(tc1[0] - c2) < tc1[1]+err2 ) ])
		if len(potential_columns)>0:
			redshifts.append(tc1[2]/CIV_1 - 1)
	return redshifts

def guess_CIV(wave,flux,err, z_threshold = 0.0005, column_threshold = 13,
				column_error_max = 0.2
				RESVEL = 30.0, window_size = 20):
	redshift_CIV_guesses = []
	# Constants
	CIV_1 = 1548.2020    
	CIV_2 = 1550.7740
	# Columns_1
	columns_1, columns_2,error_1,error_2, wave_c = logN(wave,flux,err,delta_width = 20,RESVEL = RESVEL,
														column_error_max = column_error_max, window_size=window_size)
	# Find paired densities + thresholding
	CIV_redshifts = potential_CIV_from_column_densities(wave_c,columns_1,
						columns_2, error_1,error_2, threshold_columns = column_threshold,
						tol_columns = 0.1, tol_redshift = z_threshold)

	#print(CIV_redshifts)
	plt.errorbar(wave_c, columns_1,error_1)
	plt.errorbar(wave_c, columns_2,error_2)
	plt.plot(wave,flux)
	plt.scatter((1+np.array(CIV_redshifts))*CIV_1, np.ones(len(CIV_redshifts))*1.5)
	plt.show()
	return CIV_redshifts	
	
'''
def plot_completeness():
	# Create array to store completeness
	# Loop over b,N
	# find if redshift is matching
	# count
	inputs = np.loadtxt('./Completeness/input_logN_13.1_b_20.txt')
	guesses = np.loadtxt('./Completeness/guesses_logN_13.1_b_20.txt')
	print(len(guesses), len(inputs))
'''


	



	