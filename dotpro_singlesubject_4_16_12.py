#!/bin/env python

import os                                    		# system functions
import nipype.pipeline.engine as pe          		# pypeline engine
import nipype.algorithms.modelgen as model   		# model generation
import nipype.algorithms.rapidart as ra      		# artifact detection
import nipype.interfaces.freesurfer as fs    		# freesurfer
import nipype.interfaces.spm as spm          		# spm
import nipype.interfaces.fsl as fsl			# fsl
import nipype.interfaces.fsl.maths as math   		#for dilating of the mask
import nipype.interfaces.io as nio           		# i/o routines
import nipype.interfaces.utility as util     		# utility
import nipype.interfaces.base as base        		# base routines
from nipype.interfaces.utility import IdentityInterface

from nipype.workflows.fmri.fsl import create_susan_smooth
from nipype.interfaces.ants import WarpImageMultiTransform
from nipype.interfaces.base import Bunch 		# not used
import numpy as np

subjects_list = ['SAD_C19','SAD_C20','SAD_C21','SAD_C22','SAD_C23','SAD_C24','SAD_C25','SAD_C26','SAD_C27','SAD_C28','SAD_C29','SAD_C30','SAD_C31','SAD_C32','SAD_C33','SAD_C34','SAD_C35','SAD_C36','SAD_C37','SAD_C38','SAD_C39','SAD_C40','SAD_C41','SAD_C42','SAD_P04','SAD_P05','SAD_P06','SAD_P07','SAD_P08','SAD_P09','SAD_P11','SAD_P12','SAD_P13','SAD_P14','SAD_P15','SAD_P16','SAD_P17','SAD_P18','SAD_P19','SAD_P20','SAD_P21','SAD_P22','SAD_P23','SAD_P24','SAD_P25','SAD_P26','SAD_P27','SAD_P28','SAD_P29','SAD_P30','SAD_P31']

#['SAD_P01','SAD_P02']  do not use - no  func data
#'SAD_P03','SAD_P10',, #at least one run is oriented inconsistently with the others


runs 		= ['007','008','009']
study_TR 	= 2.0
study_FWHM 	= 5


# set some paths
subjects_dir = '/Volumes/lashley/DotPro.01/Analysis/nipype/'
experiment_dir = '/Volumes/lashley/DotPro.01/'
data_dir_name = '/DotPro.01/'
T1location = experiment_dir + data_dir_name + '/Data/anat/'
T1_identifier = 'anat.nii.gz'

ANTS_template  = '/Volumes/lashley/DotPro.01/Analysis/ANTS/final_template/DOTPRO01template.nii.gz'


#~~~~~~~~~~~~~~~~~~~~~below are subject- and study-specific events.~~~~~~~~~~~~~~~~~~~~~#



def get_events(subject_id):
	
	path2events = '/Volumes/lashley/DotPro.01/Data/behav/fix_onset_iszero'
	
	event_1		= path2events+'/%(MYSUBJECT)s/007/ACL.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_2		= path2events+'/%(MYSUBJECT)s/007/ACR.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_3		= path2events+'/%(MYSUBJECT)s/007/AIL.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_4		= path2events+'/%(MYSUBJECT)s/007/AIR.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_5		= path2events+'/%(MYSUBJECT)s/007/AVL.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_6		= path2events+'/%(MYSUBJECT)s/007/AVR.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_7		= path2events+'/%(MYSUBJECT)s/007/HCL.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_8		= path2events+'/%(MYSUBJECT)s/007/HCR.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_9		= path2events+'/%(MYSUBJECT)s/007/HIL.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_10	= path2events+'/%(MYSUBJECT)s/007/HIR.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_11	= path2events+'/%(MYSUBJECT)s/007/HVL.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_12	= path2events+'/%(MYSUBJECT)s/007/HVR.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_13	= path2events+'/%(MYSUBJECT)s/007/fixation.run001.txt' %{"MYSUBJECT":(subject_id)}
	
	event_14	= path2events+'/%(MYSUBJECT)s/008/ACL.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_15	= path2events+'/%(MYSUBJECT)s/008/ACR.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_16	= path2events+'/%(MYSUBJECT)s/008/AIL.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_17	= path2events+'/%(MYSUBJECT)s/008/AIR.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_18	= path2events+'/%(MYSUBJECT)s/008/AVL.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_19	= path2events+'/%(MYSUBJECT)s/008/AVR.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_20	= path2events+'/%(MYSUBJECT)s/008/HCL.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_21	= path2events+'/%(MYSUBJECT)s/008/HCR.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_22	= path2events+'/%(MYSUBJECT)s/008/HIL.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_23	= path2events+'/%(MYSUBJECT)s/008/HIR.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_24	= path2events+'/%(MYSUBJECT)s/008/HVL.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_25	= path2events+'/%(MYSUBJECT)s/008/HVR.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_26	= path2events+'/%(MYSUBJECT)s/008/fixation.run002.txt' %{"MYSUBJECT":(subject_id)}
	
	event_27	= path2events+'/%(MYSUBJECT)s/009/ACL.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_28	= path2events+'/%(MYSUBJECT)s/009/ACR.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_29	= path2events+'/%(MYSUBJECT)s/009/AIL.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_30	= path2events+'/%(MYSUBJECT)s/009/AIR.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_31	= path2events+'/%(MYSUBJECT)s/009/AVL.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_32	= path2events+'/%(MYSUBJECT)s/009/AVR.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_33	= path2events+'/%(MYSUBJECT)s/009/HCL.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_34	= path2events+'/%(MYSUBJECT)s/009/HCR.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_35	= path2events+'/%(MYSUBJECT)s/009/HIL.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_36	= path2events+'/%(MYSUBJECT)s/009/HIR.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_37	= path2events+'/%(MYSUBJECT)s/009/HVL.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_38	= path2events+'/%(MYSUBJECT)s/009/HVR.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_39	= path2events+'/%(MYSUBJECT)s/009/fixation.run003.txt' %{"MYSUBJECT":(subject_id)}
	
	
	
	events		= [
			
			[event_1,event_2,event_3,event_4,event_5,event_6,event_7,event_8,event_9,event_10,event_11,event_12,event_13],
			[event_14,event_15,event_16,event_17,event_18,event_19,event_20,event_21,event_22,event_23,event_24,event_25,event_26],
			[event_27,event_28,event_29,event_30,event_31,event_32,event_33,event_34,event_35,event_36,event_37,event_38,event_39]
				
			]
	print events	
	return events


#~~~~~~~~~~~~~~~~~~~~~below are necessary functions~~~~~~~~~~~~~~~~~~~~~#

def pathfinder(subject, T1name):
	import os
	experiment_dir = '/Volumes/lashley/DotPro.01/Data/anat/%s/anat.nii.gz' (subject)
	return os.path.join(experiment_dir, subject, T1name)

def getthreshop(thresh):
    return ['-thr %.10f -Tmin -bin'%(0.1*val[1]) for val in thresh]

def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files

def pickmiddle(files):
    from nibabel import load
    import numpy as np
    middlevol = []
    for f in files:
        middlevol.append(int(np.ceil(load(f).get_shape()[3]/2)))
    return middlevol

def pickvol(filenames, fileidx, which):
    from nibabel import load
    import numpy as np
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filenames[fileidx]).get_shape()[3]/2))
    else:
        raise Exception('unknown value for volume selection : %s'%which)
    return idx

def getbtthresh(medianvals):
    return [0.75*val for val in medianvals]

def chooseindex(fwhm):
    if fwhm<1:
        return [0]
    else:
        return [1]

def getmeanscale(medianvals):
    return ['-mul %.10f'%(10000./val) for val in medianvals]

def getusans(x):
    return [[tuple([val[0],0.75*val[1]])] for val in x]

tolist = lambda x: [x]
highpass_operand = lambda x:'-bptf %.10f -1'%x




#~~~~~~~~~~~~~~~~~~~~~below is recon all~~~~~~~~~~~~~~~~~~~~~#


infosource = pe.Node(

		interface=util.IdentityInterface(
		
			fields				= ['subject_id'],
			),iterables 			= [('subject_id', subjects_list)],
			name				= "infosource")

#infosource.iterables.ANTS_template = ANTS_template


datasource = pe.Node(

		interface=nio.DataGrabber(
		
			base_directory			= subjects_dir,
			template 			= 'Data/anat/%s/' + T1_identifier,
			infields			= ['subject_id'],
			outfields			= ['anat']),
			name 				= 'datasource')
			
info 							= dict(	func=[['subject_id', 'subject_id']],
								struct=[['subject_id','anat']]	)
	
datasource.inputs.template_args 			= info


reconall = pe.Node(
		
		interface=fs.ReconAll(
		
			directive 			= 'all',
			subjects_dir			= subjects_dir
			),name				= "reconall")


reconflow = pe.Workflow(

		base_dir 				= subjects_dir + '/single_subject/workflows/',
		name					= "reconall")


reconflow.connect([
		
		(infosource, reconall,[('subject_id', 'subject_id')]),
		(infosource, reconall,[(('subject_id', pathfinder, T1_identifier),'T1_files')]),

		])
#execute the recon workflow
try:
	reconflow.write_graph()
	reconflow.run()
	print '==========================\n'
	print "Workflow 'RECONFLOW' Completed Normally \n"
	print "==========================\n"
	
except:
	print '==========================\n'
	print "Workflow 'RECONFLOW' Has either already been run or has terminated with Errors.\n"
	print '==========================\n'

#~~~~~~~~~~~~~~~~~~~~~below is preproc~~~~~~~~~~~~~~~~~~~~~#


preproc = pe.Workflow(
	
		base_dir 				= subjects_dir,
		name					= 'preproc')


		
datagrabber = pe.Node(

		interface=nio.DataGrabber(
		
			infields			= ['subject_id'],
			outfields			= ['func', 'anat']),
			
				base_directory		= subjects_dir,
				name 			= 'datagrabber')

datagrabber.inputs.template_args			= dict(	func	= [[ 'subject_id','run' ]],
								anat	= [[ 'subject_id' ]],
								)
								
datagrabber.inputs.template				= '*'
datagrabber.inputs.field_template 			= dict	(func = experiment_dir + '/Data/clean/%s/%s/f.nii.gz',
								 anat = experiment_dir + '/Data/anat/%s/anat.nii.gz',
								 )
								 
datagrabber.inputs.sorted 				= True
datagrabber.inputs.run					= runs


#preproc.connect(infosource,'subject_id',datagrabber,'subject_id')



inputnode = pe.Node(

		interface=util.IdentityInterface(
		
			fields				= ['subject_id','anat','func','highpass','fwhm']),
			
				name			= 'inputnode')
				
inputnode.inputs.fwhm					= study_FWHM


fssource = pe.Node(

		interface=nio.FreeSurferSource(
		
			subjects_dir 			= subjects_dir),
				
				name			='fssource')


img2float = pe.MapNode(

		interface=fsl.ImageMaths(
			
			out_data_type			= 'float',
			op_string			= '',
			suffix				= '_dtype'),
			
				iterfield		= ['in_file'],
				output_type 		= 'NIFTI',
				name			= 'img2float')


mri_convert = pe.MapNode(

		interface=fs.MRIConvert(
		
			out_type 			= 'nii'),
			
				iterfield 		= 'in_file',
				name			= "mri_convert")


slicetimer = pe.MapNode(

		interface=fsl.SliceTimer(
		
			interleaved			= True,
			time_repetition 		= study_TR,
			output_type 			= 'NIFTI',
			#time_acquisition		= ((study_TR)-(study_TR/32)),
			#slice_order			= range(32,0,-1),
			#ref_slice 			= 1,
			#num_slices 			= 32),
			),
				iterfield		= ['in_file'],
				name			= "slicetimer")


realign = pe.MapNode(

		interface=spm.Realign(),
		
				write_mask		= True,
				iterfield 		= ['in_files'],
				name 			= "realign")


func2anat = pe.MapNode(

		interface=fs.BBRegister(
		
			contrast_type 			= 't2',
			init 				= 'fsl',
			subjects_dir 			= subjects_dir,
			registered_file 		= True,
			out_fsl_file 			= True),
				
				iterfield		= ['source_file'],
				name			= 'bbregister')


fssource = pe.Node(

		interface=nio.FreeSurferSource(
			
			subjects_dir 			= subjects_dir),
			
				name			= 'fssource')


#Node: Binarize -  to binarize the aseg file for the dilation
binarize = pe.Node(

		interface=fs.Binarize(
		
			min 				= 0.5,
			out_type 			= 'nii'),
				
				name			= 'binarize')
	

#Node: DilateImage - to dilate the binarized aseg file and use it as a mask
dilate = pe.Node(

		interface=math.DilateImage(
		
			operation 			= 'max',
			output_type 			= 'NIFTI'),
			
				name			= 'dilate')
			

meanfunc = pe.MapNode(
 
 		interface=fsl.ImageMaths(
		
			op_string 			= '-Tmean', 
			suffix				= '_mean'),
					
				iterfield		= ['in_file'],
				name			= 'meanfunc')


meanfuncmask = pe.MapNode(

		interface=fsl.BET(
		
			mask 				= True, 
			no_output			= True,frac = 0.3),
			
				iterfield		= ['in_file'],
				name 			= 'meanfuncmask')


maskfunc = pe.MapNode(

		interface=fsl.ImageMaths(
		
			suffix				= '_bet',
			op_string			= '-mas'),
			
				iterfield		= ['in_file', 'in_file2'],
				name 			= 'maskfunc')


getthresh = pe.MapNode(

		interface=fsl.ImageStats(
		
			op_string			= '-p 2 -p 98'),
			
				iterfield		= ['in_file'],
				name			= 'getthreshold')


threshold = pe.MapNode(
		
		interface=fsl.ImageMaths(
		
			out_data_type			= 'char',
			suffix				= '_thresh'),
			
				iterfield		= ['in_file', 'op_string'],
				name			= 'threshold')
	

dilatemask = pe.MapNode(

		interface=fsl.ImageMaths(
		
			suffix				= '_dil',
			op_string			= '-dilF'),
				
				iterfield		= ['in_file'],
				name			= 'dilatemask')


maskfunc2 = pe.MapNode(

		interface=fsl.ImageMaths(
		
			suffix				= '_mask',
			op_string			= '-mas'),
			
				iterfield		= ['in_file', 'in_file2'],
				name			= 'maskfunc2')
				
				
susansmooth = create_susan_smooth()
susansmooth.inputs.inputnode.fwhm 			= study_FWHM

	
art = pe.MapNode(

		interface=ra.ArtifactDetect(
		
			norm_threshold   		= 1,
			zintensity_threshold 		= 3,
			mask_type			= 'file',
			parameter_source     		= 'SPM',
			use_differences 		= [True,False],
			use_norm 			= True),
			
				iterfield 		= ['realigned_files', 'realignment_parameters','mask_file'], #This might not be right - probably shouldnt iterate over masks?  Or, how can we be sure that the mask matches the func run?
				name			= "art")
				


'''

add stimulus corr! 
http://www.mit.edu/~satra/nipype-nightly/interfaces/generated/nipype.algorithms.rapidart.html

art = pe.MapNode(

		interface=ra.ArtifactDetect(
		
			use_differences = [True, False],
			use_norm = True,
			norm_threshold = 1,
			zintensity_threshold = 3,
			parameter_source = 'FSL',
			mask_type = 'file'),
			iterfield=['realigned_files', 'realignment_parameters'],
			name="art")
'''


highpass = pe.MapNode(

		interface=fsl.ImageMaths(
		
			suffix				= '_tempfilt'),
			
				iterfield		= ['in_files'],
				name			= 'highpass')


outputnode = pe.Node(

		interface=util.IdentityInterface(
		
			fields				= [	'highpassed_files',
								'min_cost_file',
								'out_fsl_file',
								'out_reg_file',
								'registered_file',
								'outlier_files',
								'smoothed_files',
								'intensity_files',
								'norm_files',
								'plot_files',
								'statistic_files',
								'mask',
								'aseg_mask',
								'mean_func',
								'subject_id',
								'ANTS_template',
								'anat',
								'masked_func_data',
								'realignment_parameters',
								'intensity_files',
								'functional_mask',
								]),
				name			= 	'outputnode')


#now, connect the nodes

preproc.connect(infosource,'subject_id',inputnode,'subject_id')
preproc.connect(infosource,'subject_id',datagrabber,'subject_id')
preproc.connect(inputnode,'subject_id',fssource,'subject_id')
preproc.connect(inputnode,'subject_id',outputnode,'subject_id')
preproc.connect(fssource, 'aseg', binarize,'in_file')
preproc.connect(dilate,'out_file',outputnode,'aseg_mask')
preproc.connect(binarize, 'binary_file', dilate, 'in_file')
preproc.connect(realign,'realigned_files', meanfunc, 'in_file')
preproc.connect(meanfunc,'out_file', meanfuncmask, 'in_file')
preproc.connect(meanfuncmask,'mask_file', maskfunc, 'in_file2')
preproc.connect(meanfuncmask,'mask_file',outputnode,'functional_mask')
preproc.connect(realign, 'realigned_files',maskfunc,'in_file')
preproc.connect(maskfunc, 'out_file', getthresh, 'in_file')
preproc.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')
preproc.connect(maskfunc, 'out_file', threshold, 'in_file')
preproc.connect(threshold, 'out_file', dilatemask, 'in_file')
preproc.connect(dilatemask, 'out_file', outputnode, 'mask')
preproc.connect(realign, 'realigned_files', maskfunc2, 'in_file')

'''
#need to add highpass filter. 
preproc.connect(inputnode, ('highpass', highpass_operand), highpass, 'op_string')
preproc.connect(realign,'realigned_files',highpass,'in_file')
preproc.connect(highpass,'out_file',maskfunc2,'in_file') 
'''

preproc.connect(dilatemask, 'out_file', maskfunc2, 'in_file2')
preproc.connect(maskfunc2, 'out_file', susansmooth, 'inputnode.in_files')
preproc.connect(maskfunc2, 'out_file', outputnode, 'masked_func_data')
preproc.connect(dilatemask, 'out_file', susansmooth, 'inputnode.mask_file')
preproc.connect(datagrabber,'func',inputnode,'func')
preproc.connect(datagrabber,'anat',inputnode,'anat')
preproc.connect(datagrabber,'anat',outputnode,'anat')
preproc.connect(inputnode,'func',img2float,'in_file')
preproc.connect(img2float,'out_file',mri_convert,'in_file')
preproc.connect(mri_convert, 'out_file', slicetimer, 'in_file')
preproc.connect(slicetimer,'slice_time_corrected_file',realign, 'in_files')
preproc.connect(realign,'realigned_files',func2anat,'source_file')
preproc.connect(realign,'realignment_parameters',outputnode,'realignment_parameters')
preproc.connect(inputnode,'subject_id',func2anat,'subject_id')

				
preproc.connect(realign,'realignment_parameters',art,'realignment_parameters')	# art
preproc.connect(realign,'realigned_files',art,'realigned_files')		# art
preproc.connect(meanfuncmask,'mask_file',art,'mask_file')			# art
preproc.connect(art,'intensity_files',outputnode,'intensity_files')		# art
#if using stimulus_corr, need to send these outputs to the outputnode

#preproc.connect(art,'outlier_files',outputnode,'outlier_files')
#preproc.connect(art,'intensity_files',outputnode,'intensity_files')
#preproc.connect(art,'norm_files',outputnode,'norm_files')
#preproc.connect(art,'plot_files',outputnode,'plot_files')
#preproc.connect(art,'statistic_files',outputnode,'statistic_files')


preproc.connect(susansmooth,'outputnode.smoothed_files', outputnode,'smoothed_files')	#susan smooth -  output
preproc.connect(func2anat,'min_cost_file',outputnode,'min_cost_file')
preproc.connect(func2anat,'out_fsl_file',outputnode,'out_fsl_file')
preproc.connect(func2anat,'out_reg_file',outputnode,'out_reg_file')
preproc.connect(func2anat,'registered_file',outputnode,'registered_file')
preproc.connect(meanfunc,'out_file',outputnode,'mean_func')








#~~~~~~~~~~~~~~~~~~~~~below is 1st level~~~~~~~~~~~~~~~~~~~~~#


#define the contrasts.  T

			
contrast1 = ('ACL>fixation','T', 	['ACL','fixation'],		[1,-1])
contrast2 = ('ACR>fixation','T', 	['ACR','fixation'],		[1,-1])
contrast3 = ('ACL_ACR>fixation','T',	['ACL','ACR','fixation'],	[0.5,0.5,-1])
contrast4 = ('HCL>fixation','T', 	['HCL','fixation'],		[1,-1])
contrast5 = ('HCR>fixation','T', 	['HCR','fixation'],		[1,-1])
contrast6 = ('HCL_HCR>fixation','T',	['HCL','HCR','fixation'],	[0.5,0.5,-1])
contrast7 = ('AVL>fixation','T',	['AVL','fixation'],		[1,-1])
contrast8 = ('AVR>fixation','T',	['AVR','fixation'],		[1,-1])
contrast9 = ('AVL_AVR>fixation','T',	['AVL','AVR','fixation'],	[0.5,0.5,-1])
contrast10= ('AIL>fixation','T',	['AIL','fixation'],		[1,-1])
contrast11= ('AIR>fixation','T',	['AIR','fixation'],		[1,-1])
contrast12= ('AIL_AIR>fixation','T',	['AIL','AIR','fixation'],	[0.5,0.5,-1])
contrast13= ('HVL>fixation','T',	['HVL','fixation'],		[1,-1])
contrast14= ('HVR>fixation','T',	['HVR','fixation'],		[1,-1])
contrast15= ('HVL_HVR>fixation','T',	['HVL','HVR','fixation'],	[0.5,0.5,-1])
contrast16= ('HIL>fixation','T',	['HIL','fixation'],		[1,-1])
contrast17= ('HIR>fixation','T',	['HIR','fixation'],		[1,-1])
contrast18= ('HIL_HIR>fixation','T',	['HIL','HIR','fixation'],	[0.5,0.5,-1])
contrast19= ('invalid>valid','T',	['HVL','HVR','AVL','AVR','HIL','HIR','AIL','AIR'],[0.25,0.25,0.25,0.25,-0.25,-0.25,-0.25,-0.25])
contrast20= ('valid>invalid','T',	['HIL','HIR','AIL','AIR','HVL','HVR','AVL','HVR'],[0.25,0.25,0.25,0.25,-0.25,-0.25,-0.25,-0.25])
contrast21= ('angryValid>angryInvalid','T',	['AVL','AVR','AIL','AIR'],[0.5,0.5,-0.5,-0.5])
contrast22= ('angryInvalid>angryValid','T',	['AIL','AIR','AVL','AVR'],[0.5,0.5,-0.5,-0.5])
contrast23= ('AVL>AVR','T',			['AVL','AVR'],		[1,-1])
contrast24= ('AVR>AVL','T',			['AVR','AVL'],		[1,-1])
contrast25= ('happyValid>happyInvalid','T',	['HVL','HVR','HIL','HIR'],[0.5,0.5,-0.5,-0.5])
contrast26= ('happyInvalid>happyValid','T',	['HIL','HIR','HVL','HVR'],[0.5,0.5,-0.5,-0.5])
contrast27= ('HVL>HVR','T',			['HVL','HVR'],		[1,-1])
contrast28= ('HVR>HVL','T',			['HVR','HVL'],		[1,-1])
contrast29= ('HVR>AVR','T',			['HVR','AVR'],		[1,-1])
contrast30= ('AVR>HVR','T',			['AVR','HVR'],		[1,-1])
contrast31= ('AIL>AIR','T',			['AIL','AIR'],		[1,-1])
contrast32= ('AIR>AIL','T',			['AIR','AIL'],		[1,-1])
contrast33= ('angryCue>happyCue','T',		['ACL','ACR','HCL','HCR'],[0.5,0.5,-0.5,-0.5])
contrast34= ('happyCue>angryCue','T',		['HCL','HCR','ACL','ACR'],[0.5,0.5,-0.5,-0.5])
contrast35= ('angryValid>happyValid','T',	['AVL','AVR','HVL','HVR'],[0.5,0.5,-0.5,-0.5])
contrast36= ('happyValid>angryValid','T',	['HVL','HVR','AVL','AVR'],[0.5,0.5,-0.5,-0.5])
contrast37= ('angryInvalid>happyInvalid','T',	['AIL','AIR','HIL','HIR'],[0.5,0.5,-0.5,-0.5])
contrast38= ('happyInvalid>angryInvalid','T',	['HIL','HIR','AIL','AIR'],[0.5,0.5,-0.5,-0.5])
contrast39 = ('All>fixation','T',
			['ACL','ACR','HCL','HCR','AVL','AVR','AIL','AIR','HVL','HVR','HIL','HIR','fixation'],
			[0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,0.0833333,-1])
contrast40 = ('fixation>All','T',
			['ACL','ACR','HCL','HCR','AVL','AVR','AIL','AIR','HVL','HVR','HIL','HIR','fixation'],
			[-0.0833333,-.0833333,-0.0833333,-0.0833333,-0.0833333,-0.0833333,-0.0833333,-0.0833333,-0.0833333,-0.0833333,-0.0833333,-0.0833333,1])

			
contrasts = [contrast1,contrast2,contrast3,contrast4,contrast5,contrast6,contrast7,contrast8,contrast9,contrast10,contrast11,contrast12,contrast13,contrast14,contrast15,contrast16,contrast17,contrast18,contrast19,contrast20,contrast21,contrast22,contrast23,contrast24,contrast25,contrast26,contrast27,contrast28,contrast29,contrast30,contrast31,contrast32,contrast33,contrast34,contrast35,contrast36,contrast37,contrast38,contrast39,contrast40]

'''

# need a more elegant way of constructing this list of objects. the loop will only construct a a list of strings. need to convert to objs
for i in range(0,39):
	
	x = 'contrast%d' %(i)
	eval(x)
	contrasts.append(x)
	
class contrasts(object):
		pass
eval("contrasts")
'''


#initiate the 1st level workflow
firstlevel_volume = pe.Workflow(
	
		base_dir 				= subjects_dir,
		name					= 'firstlevel_volume')


infosource = pe.Node(

		interface=util.IdentityInterface(
		
			fields				= ['subject_id']),
			iterables 			= ('subject_id', subjects_list),#subject_id),
			
				name			= "infosource")


inputnode = pe.Node(
	
	
		interface=util.IdentityInterface(
		
			fields				=['anat',
							'session_info',
							'bbreg_files',
							'contrasts']),
							
				name			= 'inputnode')

inputnode.inputs.contrasts = contrasts
inputnode.inputs.anat = 'anat'


convert_preproc2nii = pe.MapNode(

		interface=fs.MRIConvert(
		
			out_type 			= 'nii'),
			
				iterfield 		= 'in_file',
				name			= "convert_preproc2nii")


modelspec = pe.Node(
	
		interface = model.SpecifySPMModel(
		
			concatenate_runs 		= True,
			high_pass_filter_cutoff 	= 128,
			input_units			= 'secs',
			time_repetition 		= study_TR,
			), 
			
				name 			= 'modelspec')


level1design = pe.Node(

		interface=spm.Level1Design(
			
			interscan_interval		= study_TR,
			timing_units 			= 'secs',
			bases 				= {'hrf':{'derivs': [1,1]}} ),
			
				name			= "level1design")


level1estimate = pe.Node(

		interface=spm.EstimateModel(
		
			estimation_method 		= {'Classical' : 1},
			), name				= "level1estimate")
		

contrastestimate = pe.Node(

		interface = spm.EstimateContrast(
		
			
			), name				="contrastestimate")

#contrastestimate.inputs.use_derivs 			=True

'''
stimulus_corr= pe.Node(
	
		interface=ra.StimulusCorrelation(
		
			concatenated_design		= True,
			
			), name				="stimulus_corr")
'''


			
			

#get the warp and afflines produced by ANTS
def get_transformation_series(subject_id):
	
	image		= '/Volumes/lashley/DotPro.01/Analysis/ANTS/AUTREG01%s_brainWarp.nii.gz' %(subject_id)
	affline 	= '/Volumes/lashley/DotPro.01/Analysis/ANTS/AUTREG01%s_brainAffine.txt' %(subject_id)
	warpfiles 	= [image,affline]
	print warpfiles
	return warpfiles


#warp betas to ANTS Template
warp = pe.MapNode(
		interface = WarpImageMultiTransform(),
			
			iterfield 			= ['moving_image'],
			name				= "warp")

warp.inputs.reference_image 				= ANTS_template


mri_convert2nii = pe.MapNode(

		interface=fs.MRIConvert(
		
			out_type 			= 'nii'),
			
				iterfield 		= 'in_file',
				name			= "mri_convert2nii")


def pickBBReg(filename):
	try: 
		n = len(filename) 
		item = filename[0]  #picks the first bbreg file in the run order
		return item
		print '==========================\n'
		print item
		print '==========================\n'

	except:
        	print '==========================\n'
		print "Error - Can't pick registration2anat"
		print '==========================\n'
	
	
		
vol2vol = pe.MapNode(
		
		interface = fs.ApplyVolTransform(
		
			no_resample 			= True),
			iterfield			= ['source_file'],
			name 				= "vol2vol")
			


'''
selectcontrast = pe.Node(

		interface=util.Select(
			
			), 
			iterables 			= ('index',[[i] for i in range(len(contrasts))],
			),name				= "selectcontrast")






overlaystats = pe.Node(

		interface=fsl.Overlay(
		
			stat_thresh 			= (0.1,2),
			show_negative_stats		= True,
			auto_thresh_bg			= True,
			transparency 			= False,
			background_image 		= ANTS_template
			), name				= "overlaystats")
	
slicestats = pe.Node(
		
		interface=fsl.Slicer(
		
			all_axial 			= True,
			image_width 			= 1000,
			), name				="slicestats")
			
firstlevel_volume.connect(warp,'output_images',selectcontrast,'inlist')
firstlevel_volume.connect(selectcontrast,'out',overlaystats,'stat_image')
firstlevel_volume.connect(overlaystats,'out_file',slicestats,'in_file')
firstlevel_volume.connect(slicestats,'out_file',firstlevel_outputnode,'resliced_pngs')	
			
'''	
		
firstlevel_outputnode = pe.Node(

		interface=util.IdentityInterface(
			
			fields				=['session_info',
							'mask_image',
							'in_anat',
							'bbreg_files',
							'resliced_pngs',
							'unreg_nii',
							'warped2template']),
			name				='firstlevel_outputnode')
			



datasink = pe.Node(nio.DataSink(),
				
			name 				= "datasink")

datasink.inputs.base_directory = subjects_dir

#connect up the first level nodes
firstlevel_volume.connect(convert_preproc2nii, 'out_file', modelspec,'functional_runs')
firstlevel_volume.connect(inputnode,'contrasts',contrastestimate,'contrasts')
firstlevel_volume.connect(modelspec, 'session_info',firstlevel_outputnode,'session_info')
firstlevel_volume.connect(modelspec,'session_info',level1design,'session_info')
firstlevel_volume.connect(level1design, 'spm_mat_file', level1estimate,'spm_mat_file')
firstlevel_volume.connect(level1estimate, 'beta_images',contrastestimate, 'beta_images')
firstlevel_volume.connect(level1estimate, 'mask_image',firstlevel_outputnode, 'mask_image')
firstlevel_volume.connect(level1estimate, 'residual_image',contrastestimate, 'residual_image')
firstlevel_volume.connect(level1estimate, 'spm_mat_file',contrastestimate, 'spm_mat_file')
firstlevel_volume.connect(contrastestimate,'con_images',mri_convert2nii,'in_file')
firstlevel_volume.connect(mri_convert2nii,'out_file',vol2vol,'source_file')
firstlevel_volume.connect(mri_convert2nii,'out_file',firstlevel_outputnode,'unreg_nii')

firstlevel_volume.connect(vol2vol,'transformed_file',warp,'moving_image')
firstlevel_volume.connect(warp,'output_images',firstlevel_outputnode,'warped2template') 		#outputs from 1stlevel
firstlevel_volume.connect(vol2vol,'transformed_file',firstlevel_outputnode,'in_anat')			#outputs from 1stlevel


#sfirstlevel_volume.connect(level1design,'spm_mat_file',stimulus_corr,'spm_mat_file') 			#stimulus corr

#Clone the vol analysis and prep for surface analysis
firstlevel_surf = firstlevel_volume.clone(name='firstlevel_surface')


frameflow = pe.Workflow(

			base_dir 			= subjects_dir,
			name				= 'workflows_clean')
			
#frameflow.connect(preproc,('outputnode.intensity_files'),firstlevel_volume,('stimulus_corr.intensity_values'))  #stimulus_corr  ## this seems to be causing a problem. Should it be itereated over? Does the design indicate that its a single run? stimcor doesnt even see it now. Maybe turn this into a map node and iterate over runs
#frameflow.connect(preproc,('outputnode.realignment_parameters'),firstlevel_volume,('stimulus_corr.realignment_parameters'))

frameflow.connect(preproc,('outputnode.functional_mask'),firstlevel_volume,('firstlevel_outputnode.bbreg_files'))
frameflow.connect(preproc,('outputnode.anat'),firstlevel_volume,('vol2vol.target_file'))
frameflow.connect(preproc,('outputnode.aseg_mask'),firstlevel_volume,('level1design.mask_image'))
frameflow.connect(preproc,('outputnode.out_reg_file',pickBBReg),firstlevel_volume,('vol2vol.reg_file'))
frameflow.connect(preproc,('outputnode.subject_id',get_events),firstlevel_volume,'modelspec.event_files')
frameflow.connect(preproc,('outputnode.subject_id',get_transformation_series),firstlevel_volume,'warp.transformation_series')
frameflow.connect([(
	
			preproc,firstlevel_volume,[
			
						('outputnode.smoothed_files','convert_preproc2nii.in_file'),
						],

						) ])

#finally, the connections for surf analysis
frameflow.connect(preproc,('outputnode.anat'),firstlevel_surf,('vol2vol.target_file'))
frameflow.connect(preproc,('outputnode.out_reg_file',pickBBReg),firstlevel_surf,('vol2vol.reg_file'))
frameflow.connect(preproc,('outputnode.subject_id',get_events),firstlevel_surf,'modelspec.event_files')
frameflow.connect(preproc,('outputnode.subject_id',get_transformation_series),firstlevel_surf,'warp.transformation_series')
frameflow.connect([(
	
			preproc,firstlevel_surf,[
			
						('outputnode.masked_func_data','convert_preproc2nii.in_file'),
						],

						) ])


#make the workflow that connects all the frames
metaflow = pe.Workflow(

			base_dir 		= subjects_dir,
			name			= 'single_subject')
			

metaflow.connect([(
			frameflow,datasink,[
			
						('firstlevel_volume.firstlevel_outputnode.warped2template','2nd_level_VOL_results.@warped'),
						#('firstlevel_volume.firstlevel_outputnode.in_anat','2nd_level_VOL_results.@in_anat'),
						('firstlevel_volume.firstlevel_outputnode.bbreg_files','2nd_level_VOL_results.@bbreg'),
						('firstlevel_volume.firstlevel_outputnode.unreg_nii','2nd_level_VOL_results.@unreg_nii'),
						#('firstlevel_volume.firstlevel_outputnode.resliced_pngs','2nd_level_VOL_results.@pngs'),
						
#might need to sink the pre - resampled func cope.  Can then resample with vol2vol

						#('firstlevel_surface.firstlevel_outputnode.warped2template','2nd_level_SURF_results.@warped'),
						('firstlevel_surface.firstlevel_outputnode.in_anat','2nd_level_SURF_results.@in_anat'),
						('firstlevel_volume.firstlevel_outputnode.bbreg_files','2nd_level_SURF_results.@bbreg'),
						#('firstlevel_surface.firstlevel_outputnode.resliced_pngs','2nd_level_SURF_results.@pngs'),
						],
						) ])

# run it
try:
	metaflow.write_graph()
	metaflow.run(plugin='MultiProc', plugin_args={'n_procs' : 12})
	print
	print '==========================\n'
	print "Workflow '1ST LEVEL ANALYSIS' Completed Normally.\n"
	print "==========================\n"
	
except:
	print
	print "==========================\n"
	print "Workflow '1ST LEVEL ANALYSIS' Terminated with Errors.\n"
	print "==========================\n"