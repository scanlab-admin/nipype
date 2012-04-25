#!/bin/env python

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import linecache, os, csv, time, unicodedata, shutil, matplotlib
import numpy as np
from pylab import *
import numpy.numarray as na
from statlib import stats
from scipy.stats import f_oneway

subjects = ['SAD_C19','SAD_C20','SAD_C21','SAD_C22','SAD_C23','SAD_C24','SAD_C25','SAD_C26','SAD_C27','SAD_C28','SAD_C29','SAD_C30','SAD_C31','SAD_C32','SAD_C33','SAD_C34','SAD_C35','SAD_C36','SAD_C37','SAD_C38','SAD_C39','SAD_C40','SAD_C41','SAD_C42','SAD_P04','SAD_P05','SAD_P06','SAD_P07','SAD_P08','SAD_P09','SAD_P11','SAD_P12','SAD_P13','SAD_P14','SAD_P15','SAD_P16','SAD_P17','SAD_P20','SAD_P21','SAD_P22','SAD_P23','SAD_P24','SAD_P25','SAD_P26','SAD_P27','SAD_P28','SAD_P29','SAD_P30','SAD_P31']
#'SAD_P19',
contrasts=['01','02','04','05','07','08','10','11','13','14','16','17'] 
experiment_dir = '/Volumes/lashley/DotPro.01/Analysis/nipype/'
output = []

x= -2 #which data from the segstats file do you want to grab? This is 1 indexed, from the end.  I.e. -2 picks the 2nd to last line.

#initialize the lists
ACL= []
ACR= []
HCL= []
HCR= []
AVL= []
AVR= []
HVL= []
HVR= []
AIL= []
AIR= []
HIL= []
HIR= []

con_names = [ACL,ACR,HCL,HCR,AVL,AVR,HVL,HVR,AIL,AIR,HIL,HIR]
#iterate over contrasts
for contrast in contrasts:

    #creates header for each contrast
	contrast = str(contrast)
	output.append(['contrast:',contrast])


				
    #iterate over subjects
    	for subject in subjects:
		
		if contrast == '01':
			realname = ACL
		elif contrast == '02':
			realname = ACR
		elif contrast == '04':
			realname = HCL
		elif contrast == '05':
			realname = HCR
		elif contrast == '07':
			realname = AVL	
		elif contrast == '08':
			realname = AVR
		elif contrast == '10':
			realname = AIL		
		elif contrast == '11':
			realname = AIR		
		elif contrast == '13':
			realname = HVL		
		elif contrast == '14':
			realname = HVR			
		elif contrast == '16':
			realname = HIL			
		elif contrast == '17':
			realname = HIR
		else:
			print "I can't link the contrast to its object"
		
		#specify path to fROI datasink for each variation of segmentation
		path2anatROI = experiment_dir +'/_anat_ROIs/'
		path2Sumfile = path2anatROI + '_contrast_id_'+contrast+'_subject_id_'+subject
		statFile = path2Sumfile + '/segmentationorig/'+'/summary.stats'
	
		#extract the data from the output summary files
		dataFile = open(statFile, 'r')
		data = dataFile.readlines()
		
		pickdata= [data[x].split()[5]]

		dataFile.close()
		realname.append(pickdata)
		output.append([subject,pickdata])

    #add an empty line at the end of a contrast summary

	output.append([])
	
def mean(nums):
	if len(nums) > 0:
		return ( sum(nums) / len(nums))
	else:
		return 0.0


ACR = [item for sublist in ACR for item in sublist]
ACR = map(float,ACR)
ACR_e=stats.lsterr(ACR)
ACR_m=mean(ACR)

ACL = [item for sublist in ACL for item in sublist]
ACL = map(float,ACL)
ACL_e=stats.lsterr(ACL)
ACL_m=mean(ACL)

HCR = [item for sublist in HCR for item in sublist]
HCR = map(float,HCR)
HCR_e=stats.lsterr(HCR)
HCR_m=mean(HCR)

HCL = [item for sublist in HCL for item in sublist]
HCL = map(float,HCL)
HCL_e=stats.lsterr(HCL)
HCL_m=mean(HCL)

AVR = [item for sublist in AVR for item in sublist]
AVR = map(float,AVR)
AVR_e=stats.lsterr(AVR)
AVR_m=mean(AVR)

AVL = [item for sublist in AVL for item in sublist]
AVL = map(float,AVL)
AVL_e=stats.lsterr(AVL)
AVL_m=mean(AVL)

AIR = [item for sublist in AIR for item in sublist]
AIR = map(float,AIR)
AIR_e=stats.lsterr(AIR)
AIR_m=mean(AIR)

AIL = [item for sublist in AIL for item in sublist]
AIL = map(float,AIL)
AIL_e=stats.lsterr(AIL)
AIL_m=mean(AIL)

anova_ACL_HCL=f_oneway(ACL,HCL)
anova_ACR_HCR=f_oneway(ACR,HCR)

print '===============\n','ANOVA : ACL v HCL',anova_ACL_HCL
print '===============\n', 'ANOVA : ACR v HCR',anova_ACR_HCR

labels = ['ACR','ACL','HCR','HCL','AVR','AVL','AIR','AIL']

pe_means = [ACR_m,ACL_m,HCR_m,HCL_m,AVR_m,AVL_m,AIR_m,AIL_m]#,HVL,HVR,HIL,HIR]
error =  [ACR_e,ACL_e,HCR_e,HCL_e,AVR_e,AVL_e,AIR_e,AIL_e]
# plot data in bargraphs
xlocations = na.array(range(len(pe_means)))+0.5
width = 0.5
grid()
bar(xlocations, pe_means, yerr=error, width=width)
#yticks(range(-.1, .1))
xticks(xlocations+ width/2, labels)
xlim(0, xlocations[-1]+width*2)
title("Mean Parameter Estimate")
gca().get_xaxis().tick_bottom()
gca().get_yaxis().tick_left()
show()
#savefig('means_all.png')



#store output into a csv-file
f = open(experiment_dir+'_anat_ROI'+'_result.csv','wb')
import csv
outputFile = csv.writer(f)
for line in output:
    outputFile.writerow(line)
f.close()


