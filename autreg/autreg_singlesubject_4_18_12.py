
"""
Autism Emotion Regulation Pipeline (Nipype 0.7)
- Level 1 in subject's own functional space
- Coregister output to MNI
- ANTS normalization done offline (ANTS_batch.sh, WIMT_batch.sh)
- Level 2 using ANTS normalized con images

Created:		02-28-2013	# based on Domain pipeline script (J.A.R.)
Code Revised:		07-04-2014	
"""

import os                                    # system functions
import nipype.algorithms.modelgen as model   # model generation
import nipype.algorithms.rapidart as ra      # artifact detection
import nipype.interfaces.freesurfer as fs    # freesurfer
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.io as nio           # i/o routines
import nipype.interfaces.matlab as mlab      # how to run matlab
import nipype.interfaces.spm as spm          # spm
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.utils.filemanip import loadflat	 # some useful stuff for debugging
from nipype.interfaces.ants import WarpImageMultiTransform
import scipy.io as sio
import numpy as np
from nipype.interfaces.base import Bunch
from copy import deepcopy
import sys

#from nipype.utils.config import config
#config.enable_debug_mode()

experiment='AutReg.01'
# Tell freesurfer what subjects directory to use
experiment_dir='/Volumes/lashley/%s/'%experiment
subjects_dir = experiment_dir + 'Analysis/nipype/'
fs.FSCommand.set_default_subjects_dir(subjects_dir)

# Set the way matlab should be called
mlab.MatlabCommand.set_default_matlab_cmd("//Volumes/Macintosh_HD/Applications/MATLAB_R2011b.app//bin//matlab -nodesktop -nosplash")
#If SPM is not in your MATLAB path you should add it here
mlab.MatlabCommand.set_default_paths('/Volumes/lashley/packages/spm8/spm8/')
# Set up how FSL should write nifti files:
fsl.FSLCommand.set_default_output_type('NIFTI')


numberofruns=3
normalize='ANTS' # dartel/SPM_normalize/ANTS
datasink='off' #on/off 
test_type= "F" # "F"/"T"


if numberofruns is 4:
	subjects_list=[ 
			'20101011_11549',#C
			'20100922_11436',#C
			'20101029_11690',#C
			'20101122_11835',#C
			'20101203_11919',#C
			'20110114_12129',#C
			'20110119_12153',#C
			'20110216_12271',#C
			'20110317_12453',#C
			'20110325_12521',#C
			'20110329_12555',#C
			'20110412_12669',#C
			'20110426_12769',#C
			'20110502_12821',#C
			'20110519_12925',#C
			'20101028_11683',#A
			'20101101_11706',#A
			'20101111_11783',#A
			'20101117_11814',#A
			'20101117_11816',#A
			'20101130_11879',#A
			'20110107_12092',#A
			'20110114_12131',#A
			'20110118_12140',#A
			'20110304_12363',#A
			'20110311_12414',#A
			'20110404_12600',#A
			'20110429_12799' #A
			
			]
			
	runs 		= ['005','006','007','008']  #clean / QA checked data
	#runs 		= ['001','002','003','004']  #original data

	
T1_identifier = 'anat.nii.gz'


def get_events_4runs(subject_id):
	
	event_1_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}

	event_1_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	
	event_1_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	
	event_1_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_look.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_negative.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_positive.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_pre.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_look.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_negative.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_positive.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_pre.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/HAI_look.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/HAI_negative.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/HAI_positive.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/HAI_pre.run004.txt' %{"MYSUBJECT":(subject_id)}
	
	events		= [
			
			[event_2_run1,event_3_run1,event_4_run1,event_6_run1,event_7_run1,event_8_run1,event_10_run1,event_11_run1,event_12_run1],
			[event_2_run2,event_3_run2,event_4_run2,event_6_run2,event_7_run2,event_8_run2,event_10_run2,event_11_run2,event_12_run2],
			[event_2_run3,event_3_run3,event_4_run3,event_6_run3,event_7_run3,event_8_run3,event_10_run3,event_11_run3,event_12_run3],
			[event_2_run4,event_3_run4,event_4_run4,event_6_run4,event_7_run4,event_8_run4,event_10_run4,event_11_run4,event_12_run4],
				
			]
	print events	
	return events

def get_events_3runs(subject_id):
	

	event_1_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}

	event_1_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	
	event_1_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
		
	
	events		= [
			
			[event_2_run1,event_3_run1,event_4_run1,event_6_run1,event_7_run1,event_8_run1,event_10_run1,event_11_run1,event_12_run1],
			[event_2_run2,event_3_run2,event_4_run2,event_6_run2,event_7_run2,event_8_run2,event_10_run2,event_11_run2,event_12_run2],
			[event_2_run3,event_3_run3,event_4_run3,event_6_run3,event_7_run3,event_8_run3,event_10_run3,event_11_run3,event_12_run3],
			]
	print events	
	return events

def get_events_1runs(subject_id):
	
	event_1_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	
	event_1_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	
	event_1_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_5_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_9_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	
	events		= [
			
			[event_2_run2,event_3_run2,event_4_run2,event_6_run2,event_7_run2,event_8_run2,event_10_run2,event_11_run2,event_12_run2],
			[event_2_run3,event_3_run3,event_4_run3,event_6_run3,event_7_run3,event_8_run3,event_10_run3,event_11_run3,event_12_run3],
			[event_2_run4,event_3_run4,event_4_run4,event_6_run4,event_7_run4,event_8_run4,event_10_run4,event_11_run4,event_12_run4],
				
			]
	print events	
	return events


#initialize the pipeline
l1pipeline = pe.Workflow(name="l1pipeline")
l1pipeline.config['execution'] = {'job_finished_timeout':30}
l1pipeline.base_dir = os.path.abspath('/Volumes/lashley/AutReg.01/Analysis/nipype/')

# Map field names to individual subject runs.
infosource = pe.Node(

		interface=util.IdentityInterface(
		
			fields				= ['subject_id'],
			),
			iterables 			= [('subject_id', subjects_list)],
			name				= "infosource")



# DataGrabber node to get the input files for each subject
datasource = pe.Node(

		interface=nio.DataGrabber(
		
			infields			= ['subject_id'],
			outfields			= ['func', 'anat']),
			
				base_directory		= subjects_dir,
				name 			= 'datasource')

datasource.inputs.template_args				= dict(	func	= [[ 'subject_id','run' ]],
								anat	= [[ 'subject_id' ]],
								)
								
datasource.inputs.template				= '*'
datasource.inputs.field_template 			= dict	(func 	= experiment_dir + '/Data/func/%s/%s/f.nii.gz',
								 anat 	= experiment_dir + '/Data/anat/%s/anat.nii.gz',
								 )
								 
datasource.inputs.sorted 				= True
datasource.inputs.run					= runs

#**************
l1pipeline.connect(infosource, 'subject_id',datasource,'subject_id' )
#**************

# Slice-Time Correction Node
slice_timing = pe.MapNode(

		interface=fsl.SliceTimer(
		
			interleaved			= True,
			time_repetition 		= 1.5,
			output_type 			= 'NIFTI',
			
			),
				iterfield		= ['in_file'],
				name			= "slice_timing")

#**************
l1pipeline.connect(datasource,'func',slice_timing,'in_file')
#**************


# Motion Correction Node
realign = pe.Node(
		
		interface=spm.Realign(
		
			register_to_mean 		= True),
		 
		 		name			="realign")

#**************
l1pipeline.connect(slice_timing,'slice_time_corrected_file',realign,'in_files')
#**************


#Artifact Detection Node
art = pe.Node(interface=ra.ArtifactDetect(

			use_differences     		= [True,False],
			use_norm           		= True,
			norm_threshold    		= 1.0,
			zintensity_threshold		= 3.0,
			mask_type           		= 'file',
			parameter_source    		= 'SPM',
		),
		 
		 		name			="art")

#**************
l1pipeline.connect(realign,'realignment_parameters',art,'realignment_parameters')
l1pipeline.connect(realign,'realigned_files',art,'realigned_files')
#**************


#Stimulus correlation quality control node:
stimcor = pe.Node(interface=ra.StimulusCorrelation(), name="stimcor")
stimcor.inputs.concatenated_design = False

#**************
l1pipeline.connect(art,'intensity_files',stimcor,'intensity_values')
l1pipeline.connect(realign,'realignment_parameters',stimcor,'realignment_parameters')
#**************


# run SPM's smoothing
volsmooth = pe.Node(interface=spm.Smooth(), name="volsmooth")
volsmooth.inputs.fwhm = [5,5,5]

#**************
l1pipeline.connect(realign,'realigned_files',volsmooth,'in_files')
#**************


# Coregister node for functional images to FreeSurfer surfaces
calcSurfReg = pe.Node(interface=fs.BBRegister(),name='calcSurfReg')
calcSurfReg.inputs.init = 'fsl'
calcSurfReg.inputs.contrast_type = 't2'
calcSurfReg.inputs.registered_file = True

#**************
l1pipeline.connect(infosource,'subject_id',calcSurfReg,'subject_id')
l1pipeline.connect(realign,'mean_image',calcSurfReg,'source_file')
#**************





# Apply surface coregistration to output t-maps
applySurfRegT = pe.MapNode(interface=fs.ApplyVolTransform(),name='applySurfRegT', iterfield = ['source_file'])

#**************
l1pipeline.connect(calcSurfReg,'out_reg_file',applySurfRegT,'reg_file')
l1pipeline.connect(calcSurfReg,'registered_file',applySurfRegT,'target_file')
#**************

# Apply surface coregistration to output contrast images
applySurfRegCon = pe.MapNode(interface=fs.ApplyVolTransform(),name='applySurfRegCon', iterfield = ['source_file'])
l1pipeline.connect(calcSurfReg,'out_reg_file',applySurfRegCon,'reg_file')
l1pipeline.connect(calcSurfReg,'registered_file',applySurfRegCon,'target_file')		


# Node to find Freesurfer data
FreeSurferSource = pe.Node(interface=nio.FreeSurferSource(), name='fssource')
FreeSurferSource.inputs.subjects_dir = os.path.abspath('/Volumes/lashley/AutReg.01/Analysis/nipype/')
def get_aparc_aseg(files):
        for name in files:
            if 'aparc+aseg' in name:
                return name
        raise ValueError('aparc+aseg.mgz not found')

#**************
l1pipeline.connect(infosource,'subject_id',FreeSurferSource,'subject_id')
#**************



# Volume Transform (for making brain mask)
ApplyVolTransform = pe.Node(
			
			interface			=fs.ApplyVolTransform(),
			
				 name			='applyreg')
				 
ApplyVolTransform.inputs.inverse = True

#**************
l1pipeline.connect(realign,'mean_image',ApplyVolTransform,'source_file')
l1pipeline.connect(calcSurfReg,'out_reg_file',ApplyVolTransform,'reg_file')
l1pipeline.connect(FreeSurferSource, ('aparc_aseg', get_aparc_aseg),  ApplyVolTransform,'target_file')
#**************

# Threshold (for making brain mask)
Threshold = pe.Node(interface=fs.Binarize(dilate=1),name='threshold')
Threshold.inputs.min = 0.5
Threshold.inputs.out_type = 'nii'

#**************
l1pipeline.connect(Threshold,'binary_file',art, 'mask_file')
l1pipeline.connect(ApplyVolTransform,'transformed_file',Threshold,'in_file')
#**************


def get_aparc_aseg(files):
        for name in files:
            if 'aparc+aseg' in name:
                return name
        raise ValueError('aparc+aseg.mgz not found')



# Model Specification (NiPype) Node
modelspec = pe.Node(interface=model.SpecifyModel(), name="modelspec", overwrite=True)
modelspec.inputs.input_units = 'secs'
modelspec.inputs.time_repetition = 1.5
modelspec.inputs.high_pass_filter_cutoff = 160 #160 OR np.inf #inf because of linear / quad regressors - otherwise ~160

if numberofruns is 4:
	l1pipeline.connect(infosource, ('subject_id', get_events_4runs),modelspec,'event_files')
if numberofruns is 3:
	l1pipeline.connect(infosource, ('subject_id', get_events_3runs),modelspec,'event_files')
if numberofruns is 1:
	l1pipeline.connect(infosource, ('subject_id', get_events_1runs),modelspec,'event_files')


#**************
l1pipeline.connect(realign,'realignment_parameters',modelspec,'realignment_parameters')
l1pipeline.connect(volsmooth,'smoothed_files',modelspec,'functional_runs')
l1pipeline.connect(art,'outlier_files',modelspec,'outlier_files')
#**************



# Level 1 Design (SPM) Node
level1design = pe.Node(interface=spm.Level1Design(), name= "level1design")
level1design.inputs.timing_units = 'secs'
level1design.inputs.interscan_interval = modelspec.inputs.time_repetition
level1design.inputs.bases = {'hrf':{'derivs':[0,0]}}
level1design.inputs.model_serial_correlations = 'AR(1)' #'none'

'''
multipleRegDes.inputs.covariates = [dict(vector=[-0.30,0.52,1.75],
                                         name='nameOfRegressor1'),
                                    dict(vector=[1.55,-1.80,0.77],
                                         name='nameOfRegressor2')]
'''
#**************
l1pipeline.connect(modelspec,'session_info',level1design,'session_info')
l1pipeline.connect(Threshold,'binary_file',level1design,'mask_image')
l1pipeline.connect(level1design,'spm_mat_file',stimcor,'spm_mat_file')
#**************



# Level 1 Estimation node
level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical' : 1}
#**************
l1pipeline.connect(level1design,'spm_mat_file',level1estimate,'spm_mat_file')
#**************


# Constrast Estimation node
contrastestimate = pe.Node(

		interface		 	= spm.EstimateContrast(),
			
				name		="contrastestimate")

contrast1 = ('facepre>anyreg','T',              ['FACE_pre','FACE_positive','FACE_negative'],		[1,-0.5,-0.5])
contrast2 = ('anyreg>facepre','T',              ['FACE_pre','FACE_positive','FACE_negative'],		[-1,0.5,0.5])
contrast3 = ('faceneg>facepos','T',             ['FACE_negative','FACE_positive'],                  	[1,-1])
contrast4 = ('facepos>faceneg','T',             ['FACE_positive','FACE_negative'],                  	[1,-1])
contrast5 = ('facepre>facepos','T',             ['FACE_pre','FACE_positive'],                       	[1,-1])
contrast6 = ('facepos>facepre','T',             ['FACE_positive','FACE_pre'],                       	[1,-1])
contrast7 = ('faceneg>facepre','T',             ['FACE_negative','FACE_pre'],                       	[1,-1])
contrast8 = ('facepre>faceneg','T',             ['FACE_negative','FACE_pre'],                       	[-1,1])


#main effects
contrast35 = ('FACE_pre','T',        	['FACE_pre'],          		[1]              )
contrast36 = ('FACE_negative','T', 	['FACE_negative'],     		[1]              )
contrast37 = ('FACE_positive','T', 	['FACE_positive'],         	[1]              )

#verify the effect of the dummy variable. 
contrast44 = ('faceprevfacepre','T',             ['FACE_pre','FACE_pre'],		[0,1])
contrast45 = ('faceprevfaceneg','T',             ['FACE_pre','FACE_negative'],		[0,1])
contrast46 = ('faceprevfacepos','T',             ['FACE_pre','FACE_positive'],		[0,1])


if test_type is "F":

	contrast35 = ('FACE_pre','T',        	['FACE_pre'],          		[1]              )
	contrast36 = ('FACE_negative','T', 	['FACE_negative'],     		[1]              )
	contrast37 = ('FACE_positive','T', 	['FACE_positive'],         	[1]              )

	contrast48 = ('f_test_ANYREG_FACE','F', [contrast35, contrast36,contrast37]		 )

	contrasts = [contrast35,contrast36,contrast37,contrast48]

if test_type is "T":
	
	contrasts = [contrast1,contrast2,contrast3,contrast4,contrast5,contrast6,contrast7,contrast8]


	
contrastestimate.inputs.contrasts 		=  contrasts


#**************
l1pipeline.connect(level1estimate,'spm_mat_file',contrastestimate,'spm_mat_file')
l1pipeline.connect(level1estimate,'beta_images',contrastestimate,'beta_images')
l1pipeline.connect(level1estimate,'residual_image',contrastestimate,'residual_image')
#**************


# Have a node that converts spm TSTAT IMG files to NIFTI files so FreeSurfer doesn't have a stupid header error.
makeImgNiiT = pe.MapNode(interface=fs.MRIConvert(),name='makeImgNiiT', iterfield=['in_file'])
makeImgNiiT.inputs.in_type = 'nifti1'
makeImgNiiT.inputs.out_type = 'niigz'

# Have a node that converts spm CON (i.e. cope) IMG files to NIFTI files so FreeSurfer doesn't have a stupid header error.
makeImgNiiCon = pe.MapNode(interface=fs.MRIConvert(),name='makeImgNiiCon', iterfield=['in_file'])
makeImgNiiCon.inputs.in_type = 'nifti1'
makeImgNiiCon.inputs.out_type = 'niigz'	

if normalize is 'SPM_normalize':

	normalize_T = pe.MapNode(
		
			interface			=spm.Normalize(
			
				template		='/Volumes/lashley/packages/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
					),
					
					iterfield	= ['source'],
					name		='normalize_T')
					
	normalize_T.inputs.template			= '/Volumes/lashley/packages/fsl/data/standard/MNI152_T1_1mm_brain.nii'
	
	normalize_cons = pe.MapNode(
		
			interface			=spm.Normalize(
			
				template		='/Volumes/lashley/packages/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
					),
					
					iterfield	= ['source'],
					name		='normalize_con')
						
	normalize_cons.inputs.template			= '/Volumes/lashley/packages/fsl/data/standard/MNI152_T1_1mm_brain.nii'


	#**************
	l1pipeline.connect(contrastestimate,'spmT_images',normalize_T,'source')
	l1pipeline.connect(contrastestimate,'con_images',normalize_cons,'source')
	l1pipeline.connect(normalize_T,'normalized_source',makeImgNiiT,'in_file')
	l1pipeline.connect(normalize_cons,'normalized_source',makeImgNiiCon,'in_file')
	l1pipeline.connect(makeImgNiiCon,'out_file',applySurfRegCon,'source_file')
	l1pipeline.connect(makeImgNiiT,'out_file',applySurfRegT,'source_file')
	#**************

if normalize is 'ANTS':
	
	def get_transformation_series(subject_id):
	
		image					= '/Volumes/lashley/AutReg.01/Analysis/ANTS/%s_brainWarp.nii.gz' %(subject_id)
		affline 				= '/Volumes/lashley/AutReg.01/Analysis/ANTS/%s_brainAffine.txt' %(subject_id)
		warpfiles 				= [image,affline]
		print warpfiles
		return warpfiles
	
	#warp to ANTS Template
	warp_T = pe.MapNode(
		interface 				= WarpImageMultiTransform(
		
			reference_image 		= '/Volumes/lashley/%s/Analysis/ANTS/MNI152_T1_1mm_brain.nii.gz'%experiment,
				),
				
				iterfield 		= ['moving_image'],
				name			= "warp_T")
	warp_con = pe.MapNode(
		interface 				= WarpImageMultiTransform(
		
			reference_image 		= '/Volumes/lashley/%s/Analysis/ANTS/MNI152_T1_1mm_brain.nii.gz'%experiment,
				),
				
				iterfield 		= ['moving_image'],
				name			= "warp_con")
	#**************
	#get transformation series
	l1pipeline.connect([
		
		(infosource, warp_T,[(('subject_id', get_transformation_series),'transformation_series')]),
		(infosource, warp_con,[(('subject_id', get_transformation_series),'transformation_series')]),	])
	
	if test_type is "T":
		l1pipeline.connect(contrastestimate,'spmT_images',warp_T,'moving_image')
		l1pipeline.connect(contrastestimate,'con_images',warp_con,'moving_image') 

	if test_type is "F":
		l1pipeline.connect(contrastestimate,'spmF_images',warp_T,'moving_image') # if using F images
		l1pipeline.connect(contrastestimate,'con_images',warp_con,'moving_image') # if using F images

		
	l1pipeline.connect(warp_T,'output_image',makeImgNiiT,'in_file')
	l1pipeline.connect(warp_con,'output_image',makeImgNiiCon,'in_file')
	l1pipeline.connect(makeImgNiiCon,'out_file',applySurfRegCon,'source_file')
	l1pipeline.connect(makeImgNiiT,'out_file',applySurfRegT,'source_file')			
	#**************




# Datasink node for saving output of the pipeline
datasink = pe.Node(

		interface			=nio.DataSink(
		
			base_directory 		= os.path.abspath('/Volumes/lashley/AutReg.01/Analysis/nipype/l1output'),),
		
				name		="datasink")
				
				

def getsubs(subject_id,contrast_list):
	subs = [('_subject_id_%s/'%subject_id,'')]
	for i in range(len(contrast_list),0,-1):
		subs.append(('_applySurfRegCon%d/'%(i-1),''))
		subs.append(('_applySurfRegT%d/'%(i-1),''))
		subs.append(('_applySurfRegVar%d/'%(i-1),''))
		subs.append(('con_%04d_out_maths_warped'%(i),'var_%04d_out_warped'%(i)))
	return subs


#connections for the datasink
if datasink is 'on': 
	l1pipeline.connect([
	(infosource,datasink,[('subject_id','container'),
						
						(
						
						('subject_id',getsubs,contrastestimate.inputs.contrasts),'substitutions')]),
						(FreeSurferSource,datasink,[('brain','subj_anat.@brain')]),
						(realign,datasink,[('mean_image','subj_anat.@mean')]),
						(calcSurfReg,datasink,[('out_reg_file','surfreg'),
												('min_cost_file','qc_bbreg'),
												('registered_file','subj_anat.@reg_mean')]),
						(warp_T,datasink,[('output_image','reg_cons')]),
						(warp_con,datasink,[('output_image','reg_cons')]),
						(level1estimate,datasink,[
													('spm_mat_file','model.@spm'),
													('mask_image','model.@mask'),
													('residual_image','model.@res'),
													('RPVimage','model.@rpv')]),
						(art,datasink,[('outlier_files','qc_art.@outliers'),
										('plot_files','qc_art.@motionplots'),
										('statistic_files','qc_art.@statfiles'),
										]),
						(stimcor,datasink,[('stimcorr_files','qc_stimcor')]),
						])


l1pipeline.write_graph()
l1pipeline.run(plugin='MultiProc', plugin_args={'n_procs' : 12})
