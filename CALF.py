import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy import constants
from utils import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

############ user parameters ##################
logN_max = 13.2
logN_min = 13.2
logN_steps = 1

b_values = [20]

N_CIV_per_strength = 100
N_CIV_per_QSO = 1

window_size =  30
z_threshold = 0.001
SNR_threshold = 5
chi2_threshold = 3

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

###############################################

all_files = [f for f in listdir('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/QSOfiles_vpfit/') 
	if isfile(join('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/QSOfiles_vpfit/', f))]
files_spectra = [f for f in all_files if f[-16:-4] == '_continuum_X' and f[-4::]=='.txt']

#files_spectra =files_spectra[::-1]

# Fix strength bins, number of CIV per bin, divide by number of QSOs sightlines
logN = np.linspace(logN_min,logN_max,logN_steps)
number_loops_QSO = np.ceil(N_CIV_per_strength/len(files_spectra))

# For each QSO sightlines, for n batches of m CIV

Completeness = np.zeros(len(logN))

for b in b_values:
	for strength in logN:
		matching = 0
		input_redshifts = []
		guesses_CIV_file = open('./Completeness/guesses_logN_' + str(strength) + '_b_' +str(b)+'.txt','w')
		for loop in range(int(number_loops_QSO)):
			for i in range(len(files_spectra)):
				print(i + loop*len(files_spectra),files_spectra[i])
				z_QSO = QSO_redshift[files_spectra[i][0:10]]	
				spectra = np.loadtxt('../QSOfiles_vpfit/' +  files_spectra[i] ,skiprows =2)
				# Get RESVEL
				file = open('../QSOfiles_vpfit/' +  files_spectra[i] , 'r')
				lines= file.readlines()
				if lines[0][2:7] == 'HIRES':
					RESVEL = float(lines[1][0:-1][7::])*5
				else:
					RESVEL = float(lines[1][0:-1][7::])
				# Divide by continuum, get RESVEL
				wave = np.array([w for w,f in zip(spectra[:,0],spectra[:,1]) if f != 0.0])
				flux = np.array([f for f in spectra[:,1] if f != 0.0])
				err = np.array([e for e,f in zip(spectra[:,2],spectra[:,1]) if f != 0.0])
				continuum = np.array([c for c,f in zip(spectra[:,3],spectra[:,1]) if f != 0.0])
				flux = flux/continuum
				err = err/continuum
				dvperpix = 1e-3*constants.c*np.array([(wave[i]-wave[i-1])/wave[i] for i in range(len(wave))])
				dvperpix[0] = dvperpix[1]
				# Draw random redshifts and fixed strengths from bins
				#strengths = logN[loop*N_CIV_per_QSO_per_b:np.min((logN_steps,(loop+1)*N_CIV_per_QSO_per_b))]
				max_noise_index = np.max(np.where(err < 0.2)) 
				#print(wave[max_noise_index] / (1+z_QSO)), np.min((np.max(wave)/(CIV_2+10)-1, z_QSO))
				redshift = np.random.uniform((1+z_QSO)*1026./1216. - 1 , np.min((np.max(wave[max_noise_index])/(CIV_2+10)-1, z_QSO)),1)
				amplitude = logN_to_amp(strength,b,(1+redshift)*CIV_1)
				# Create modified flux
				mod_flux = add_CIV(wave,flux,redshift,amplitude,b/np.mean(dvperpix))
				print( 'Find z=', redshift, ' at ' , (1+redshift)*CIV_1)
				# Get CIV_guesses
				redshifts_guesses = guess_CIV(wave,mod_flux,err, SNR_threshold = SNR_threshold, 
											  z_threshold = z_threshold, chi2_threshold = chi2_threshold,
											  RESVEL = np.mean(dvperpix), window_size = window_size)
				
				input_redshifts.append(redshift)
				print(np.round(redshift,2) in np.round(redshifts_guesses,2))
				matching += np.round(redshift,2) in np.round(redshifts_guesses,2)
				guesses_CIV_file.write(str(redshifts_guesses)[1:-1] + '\n')
				# Write mock CIV redshifts in file A
		np.savetxt('./Completeness/input_logN_' + str(strength) + '_b_' +str(b)+'.txt',np.array(input_redshifts))
		Completeness[np.where(logN == strength)] = matching /(number_loops_QSO*len(files_spectra))
		print(Completeness[np.where(logN == strength)])
np.savetxt('./completeness_'+str(logN_min)+'_' + str(logN_max)+'.txt', Completeness)


### dvperpix in km/s
#dvperpix = 1e-3*constants.c*np.array([(wave[i]-wave[i-1])/wave[i] for i in range(len(wave))])
#dvperpix[0] = dvperpix[1]

# LogN approx

#doublets_full = np.array([[doublet[0],doublet[1],doublet[2],doublet[3],doublet[4]
#	, np.log10(-np.sum(np.log10(voigt_approx(wave,doublet[0],doublet[1],doublet[2],doublet[3])+1) * dvperpix)
#		/ (f_CIV_1548 * CIV_1 * column_conversion_const)) ]
#	for doublet in voigt_params])

#if len(doublets)>0:
#	print(doublets[:,1]/CIV_1-1.0)
#	print(doublets[:,2])
#	print(doublets[:,3])
#	print(doublets[:,4])
#	print(doublets[:,5])

#plt.plot(wave[lya_index::],f[lya_index::],'k')
#plt.plot(wave[lya_index::],med[lya_index::],'--b')
#plt.plot(wave[lya_index::],error[lya_index::], '--r')
#plt.vlines(x = peaks, ymin= 0, ymax = 2, color = 'k')
#plt.vlines(x =  doublets[:,1], ymin=0,ymax=3.5, color = 'r')
#plt.vlines(x =  doublets[:,1]/CIV_1*CIV_2, ymin=0,ymax=3, color = 'g')
#plt.show()



