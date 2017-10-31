import numpy as np
from scipy.optimize import curve_fit
from utils import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

QSO_redshift = {'J0148+0600': 5.923 , 'J0836+0054':5.81 , 'J0927+2001': 5.772, 
	'J1030+0524': 6.28, 'J1306+0356':6.016, 'J1319+0959': 6.132 , 'J0002+2550': 5.8 ,
	'J0050+3445': 6.25, 'J0100+2802': 6.3 , 'J0353+0104': 6.072 , 'J0818+1722': 6.0 , 
	'J0842+1218': 6.069, 'J1048+4637': 6.198 , 'J1137+3549': 6.01 , 'J1148+5251': 6.419 ,
	'J1509-1749': 6.12, 'J1602+4228': 6.09, 'J2054-0005': 6.062, 'J2315-0023': 6.117 }


all_files = [f for f in listdir('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/') 
	if isfile(join('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/', f))]
files_spectra = [f for f in all_files if f[11:14]!='gas' and f[-4::]=='.txt']
#files_gas = [f for f in all_files if f[11:14]=='gas' and f[-4::]=='.txt']


#for file_spectra in files_spectra:

file_spectra = files_spectra[9]
print file_spectra

z_QSO = QSO_redshift[file_spectra[0:10]]	
spectra = np.loadtxt('../' + file_spectra ,skiprows =1)

lya_index = np.min(np.where(spectra[:,0]>1260*(1+z_QSO)))

################ Truncate flux, error, wave above LyA #####################
wave = spectra[lya_index::,0]
f = spectra[lya_index::,1]
error = spectra[lya_index::,2]

flux = f - running_median(f,100)

kernel = gaussian(-0.5, 10,3, np.linspace(0,20,20))

SNR = matched_filter(kernel,flux,error)

peaks = np.array([w for w,s in zip(wave,SNR) if s > 5])

pairs = find_potential_doublets(peaks,1548,1550,0.001)

print np.unique(np.around(pairs,3))

plt.plot(wave,f,'k')
plt.plot(wave,running_median(f,100),'--b')
plt.vlines(x =  (np.unique(np.around(pairs,3))+1)*1548, ymin=-0.2,ymax=5, color = 'r')
plt.vlines(x =  (np.unique(np.around(pairs,3))+1)*1550, ymin=-0.2,ymax=5, color = 'g')
plt.show()







