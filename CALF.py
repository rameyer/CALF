import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy import constants
from utils import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.signal import argrelmax 


############ user parameters ##################
window_size =  100
z_threshold = 0.0005
#### Must be < 0.3 if no error profile measure (log10(2)=0.3)
column_threshold = 0.3
SNR_threshold = 3
chi2_spline = 1.3
N_sigma_detection = 2

############# Constants #######################
f_CIV_1548 = 0.194000
f_CIV_1550 = 0.097000

column_conversion_const = 2.654e-15

CIV_1 = 1548.2020    
CIV_2 = 1550.7740 

QSO_redshift = {'J0148+0600': 5.923 , 'J0836+0054':5.81 , 'J0927+2001': 5.772, 
	'J1030+0524': 6.28, 'J1306+0356':6.016, 'J1319+0959': 6.132 , 'J0002+2550': 5.8 ,
	'J0050+3445': 6.25, 'J0100+2802': 6.3 , 'J0353+0104': 6.072 , 'J0818+1722': 6.0 , 
	'J0842+1218': 6.069, 'J1048+4637': 6.198 , 'J1137+3549': 6.01 , 'J1148+5251': 6.419 ,
	'J1509-1749': 6.12, 'J1602+4228': 6.09, 'J2054-0005': 6.062, 'J2315-0023': 6.117 }

all_files = [f for f in listdir('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/') 
	if isfile(join('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/', f))]
files_spectra = [f for f in all_files if f[11:14]!='gas' and f[-4::]=='.txt']

#for file_spectra in files_spectra:

file_spectra = files_spectra[13]
print(file_spectra)

z_QSO = QSO_redshift[file_spectra[0:10]]	
spectra = np.loadtxt('../' + file_spectra ,skiprows =1)

lya_index = np.min(np.where(spectra[:,0]>1260*(1+z_QSO)))

################ Truncate flux, error, wave above LyA #####################
wave = spectra[:,0]
med = running_median(spectra[:,1],200)
f = spectra[:,1]
error = spectra[:,2]

flux = f - med

error = np.array([e if not(f==0) else np.inf for e,f in zip(error,f) ])

### dvperpix in km/s
dvperpix = 1e-3*constants.c*np.array([(wave[i]-wave[i-1])/wave[i] for i in range(len(wave))])
dvperpix[0] = dvperpix[1]

######################### Gaussian-matched filter  ###########################

kernel = gaussian( np.linspace(0,20,20),-0.5, 10,3,0)

SNR = matched_filter(kernel,flux[lya_index::],error[lya_index::])

#peaks = np.array([w for w,s in zip(wave[lya_index::],SNR) if s > SNR_threshold ])
peaks_indices = argrelmax(SNR*(SNR>SNR_threshold), order = 5)
peaks = wave[lya_index::][peaks_indices]

continuum_indices = np.where(SNR < 0.5)

spline = UnivariateSpline(wave[lya_index::][continuum_indices],f[lya_index::][continuum_indices]
	, w = 1./(error[lya_index::][continuum_indices])**2
	, k = 3, s = np.sum(1./(error[lya_index::][continuum_indices])**2)*chi2_spline )


plt.plot(wave[lya_index::],f[lya_index::],'k')
plt.plot(wave[lya_index::],spline(wave[lya_index::]), '--c')
#plt.plot(wave[lya_index::], SNR,'--g')
plt.show()

f = f / spline(wave)
error = error /spline(wave)

######################### Fitting gaussians  ###########################

### holder for Gaussian parameters
gaussian_params = np.zeros((len(peaks),4))
#### x_0, amp, fhwm_L , fwhm_G
#voigt_params = np.zeros(len(peaks),4)


for i in range(len(peaks)):
	peak = peaks[i]
	index_peak = np.where(wave==peak)[0][0]
	
	#print(-0.2,peak,1,med[index_peak])

	gaussian_params[i], _  = curve_fit(gaussian,xdata=wave[int(index_peak-window_size*0.5):int(index_peak+window_size*0.5)]
		, ydata=f[int(index_peak-window_size*0.5):int(index_peak+window_size*0.5)] 
		, sigma = error[int(index_peak-window_size*0.5):int(index_peak+window_size*0.5)]
		, p0 =[-0.2,peak,1,1] , bounds = ([-1,peak-5,0.1,1-0.000001],[0,peak+5,3,1]) , method = 'trf')
	#plt.plot(wave[lya_index::],voigt_approx(wave[lya_index::], *gaussian_params[i]))

#print(gaussian_params)

###################### Selection of doublets ########################## 

pairs_indices = find_potential_doublets(gaussian_params,CIV_1,CIV_2,z_threshold)

for i in pairs_indices:
	plt.plot(wave[lya_index::],gaussian(wave[lya_index::], *gaussian_params[i[0]]))
	plt.plot(wave[lya_index::],gaussian(wave[lya_index::], *gaussian_params[i[1]]))

pairs_column_densities = np.array([ [gaussian_params[ind[0],0], gaussian_params[ind[1],0],gaussian_params[ind[0],1] 
	, gaussian_params[ind[1],1], gaussian_params[ind[0],2], gaussian_params[ind[1],2]
	, np.log10(-np.sum(np.log10(gaussian(wave,gaussian_params[ind[0],0],gaussian_params[ind[0],1],gaussian_params[ind[0],2],gaussian_params[ind[0],3])/gaussian_params[ind[0],3]) * dvperpix)/ (f_CIV_1548 * CIV_1 * column_conversion_const))
	, np.log10(-np.sum(np.log10(gaussian(wave,gaussian_params[ind[1],0],gaussian_params[ind[1],1],gaussian_params[ind[1],2],gaussian_params[ind[1],3])/gaussian_params[ind[1],3]) * dvperpix) / (f_CIV_1550 * CIV_2 * column_conversion_const))]  
	for ind in pairs_indices])

# Apply column density threshold: should be same for CIV 1548 or CIV 1550
pairs_column_densities = np.array([x for x in pairs_column_densities if np.abs(x[6] - x[7]) < column_threshold])
# If 1 gaussian can be either CIV 1548 or CIV 1550, prefer CIV 1548
pairs_column_densities = np.array([x for x in pairs_column_densities if not(x[2] in pairs_column_densities[:,3]) ])
# Apply N sigma thresholding on CIV 1548 detection
pairs_column_densities = np.array([x for x in pairs_column_densities if np.abs(x[0]) > N_sigma_detection * error[np.min(np.where(wave>x[2]))] ] )
print(pairs_column_densities)
if len(pairs_column_densities)>0:
	print(pairs_column_densities[:,2]/CIV_1-1.0)
	print(pairs_column_densities[:,6])
	print(pairs_column_densities[:,6]-pairs_column_densities[:,7])

###################### Selection of doublets ########################## 

plt.plot(wave[lya_index::],f[lya_index::],'k')
#plt.plot(wave[lya_index::],med[lya_index::],'--b')
plt.plot(wave[lya_index::],error[lya_index::], '--r')
#plt.vlines(x = peaks, ymin= 0, ymax = 2, color = 'k')
plt.vlines(x =  pairs_column_densities[:,2], ymin=0,ymax=3.5, color = 'r')
plt.vlines(x =  pairs_column_densities[:,3], ymin=0,ymax=3, color = 'g')
plt.show()



