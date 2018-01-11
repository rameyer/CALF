import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy import constants
from utils import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import argparse

##### PARSER #####
parser = argparse.ArgumentParser(description='Input Parameters Integers.')
parser.add_argument('doppler', metavar='B', type=float,help='doppler')
parser.add_argument('logN', metavar='N', type=float,help='density')
parser.add_argument('inst', metavar='I', type=str , help='instrument_name')
args = parser.parse_args()
############ user parameters ##################
logN_max = args.logN
logN_min = args.logN
logN_steps = 1

b_values = [args.doppler]

N_CIV_per_strength = 20
N_CIV_per_QSO = 1

if args.inst == 'X': 
	column_error_max = 0.05
elif args.inst =='H': 
	column_error_max = 0.15
elif args.inst =='E': 
	column_error_max = 0.25
column_threshold = 12.0
window_size =  50

purity = False

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
# Uncomment for local use
all_files = [f for f in listdir('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/QSOfiles_vpfit/') 
	if isfile(join('/Users/RomainMeyer/Dropbox/PhD/CIVLya_Corr_QSO/QSOfiles_vpfit/', f))]
# Uncomment for legion use
#all_files = [f for f in listdir('./spectra_files/') 
#	if isfile(join('./spectra_files/', f))]
files_spectra = [f for f in all_files if f[-16:-4] == '_continuum_'+args.inst and f[-4::]=='.txt']

#files_spectra =files_spectra[::-1]

#Fix strength bins, number of CIV per bin, divide by number of QSOs sightlines
logN = np.linspace(logN_min,logN_max,logN_steps)
number_loops_QSO = np.ceil(N_CIV_per_strength/len(files_spectra))

# For each QSO sightlines, for n batches of m CIV

Completeness = np.zeros(len(logN))

for b in b_values:
	for strength in logN:
		matching = 0
		input_redshifts = []
		#guesses_CIV_file = open('./guesses_N_'+str(args.logN) + '_b_' + str(args.doppler) + '_I_' + args.inst+'.txt','w')
		for loop in range(int(number_loops_QSO)):
			for i in range(0,len(files_spectra)):
				#print(i + loop*len(files_spectra),files_spectra[i])
				print (files_spectra[i])
				z_QSO = QSO_redshift[files_spectra[i][0:10]]	
				spectra = np.loadtxt('../QSOfiles_vpfit/' +  files_spectra[i] ,skiprows =2)
				# Get RESVEL
				file = open('../QSOfiles_vpfit/' +  files_spectra[i] , 'r')
				lines= file.readlines()
				RESVEL = float(lines[1][0:-1][7::])
				# Divide by continuum
				wave = np.array([w for w,f in zip(spectra[:,0],spectra[:,1]) if f != 0.0])
				flux = np.array([f for f in spectra[:,1] if f != 0.0])
				err = np.array([e for e,f in zip(spectra[:,2],spectra[:,1]) if f != 0.0])
				continuum = np.array([c for c,f in zip(spectra[:,3],spectra[:,1]) if f != 0.0])
				flux = flux/continuum
				err = err/continuum
				# Draw random redshifts and fixed strengths from bins
				max_noise_indices = np.where(err < 0.3)
				list_CIV = []
				#for i in range(1000):
				redshift = np.random.uniform((1+z_QSO)*1026./1216. - 1 , np.min((np.max(wave[max_noise_indices])/(CIV_2+10)-1, z_QSO)),1)
				while (wave[max_noise_indices][np.min(np.where((1+redshift)*(CIV_2+0.5)<wave[max_noise_indices]))] - (1+redshift)*(CIV_2+0.5) > 1) or ((1+redshift)*(CIV_1-0.5) -  wave[max_noise_indices][np.max(np.where(((1+redshift)*(CIV_1-0.5))>wave[max_noise_indices]))] > 1) :
					redshift = np.random.uniform((1+z_QSO)*1026./1216. - 1 , np.min((np.max(wave[max_noise_indices])/(CIV_2+10)-1, z_QSO)),1)
				#	list_CIV.append(redshift)
				#plt.scatter((1+np.array(list_CIV))*CIV_1, np.ones(1000)*1.5)
				#plt.plot(wave,flux)
				#plt.show()
				#redshift = 8310/CIV_1-1
				# Create modified flux
				#mod_flux = flux+doublet_CIV_voigt(wave=wave,logN=strength,b=b,z=redshift,RESVEL = RESVEL,upsampling=20) 
				mod_flux = flux
				#print( 'Find z=', redshift, ' at ' , (1+redshift)*CIV_1)
				# Get CIV_guesses
				redshifts_guesses = guess_CIV(wave,mod_flux,err, column_threshold = column_threshold,
											  column_error_max = column_error_max,
											  RESVEL = RESVEL, window_size = window_size)
				input_redshifts.append(redshift)
				#print(np.round(redshift,2) in np.round(redshifts_guesses,2))
				matching += np.round(redshift,2) in np.round(redshifts_guesses,2)
				#guesses_CIV_file.write(str(redshifts_guesses)[1:-1] + '\n')
				# Write mock CIV redshifts in file A
		#np.savetxt('./inputs_N_'+str(args.logN) + '_b_' + str(args.doppler) + '_I_' + args.inst+'.txt',np.array(input_redshifts))
		Completeness[np.where(logN == strength)] = matching /(number_loops_QSO*len(files_spectra))
		#print(Completeness[np.where(logN == strength)])
if purity:
	np.savetxt('./purity_N_'+str(args.logN) + '_b_' + str(args.doppler) + '_I_' + args.inst+'.txt', Completeness)
else:
	np.savetxt('./completeness_N_'+str(args.logN) + '_b_' + str(args.doppler) + '_I_' + args.inst+'.txt', Completeness)




