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

subjects_list =['20101011_11549','20100922_11436','20101028_11683','20101029_11690','20101101_11706','20101111_11783','20101117_11814','20101117_11816','20101122_11835','20101130_11879','20101203_11919','20110107_12092','20110114_12129','20110114_12131','20110118_12140','20110119_12153','20110216_12271','20110304_12363','20110311_12414','20110317_12453','20110325_12521','20110329_12555','20110404_12600','20110412_12669','20110426_12769','20110429_12799','20110502_12821']


#problematic subjects

#3 runs only
#['20101116_11806','20110224_12334','20101006_11510','20110310_12406']
#no stf for run 1 ['20110519_12925','20110315_12437']
#do not use ['20101214_11986']


runs 		= ['005','006','007','008'] #clean data
#runs 		= ['001','002','003','004'] #original data
study_TR 	= 1.5
study_FWHM 	= 5


# set some paths
subjects_dir = '/Volumes/lashley/AutReg.01/Analysis/nipype/'
experiment_dir = '/Volumes/lashley/AutReg.01/'
data_dir_name = '/AutReg.01/'
T1location = experiment_dir + data_dir_name + '/Data/anat/'
T1_identifier = 'anat.nii.gz'
ANTS_template  = '/Volumes/lashley/AutReg.01/Analysis/ANTS/final_template/AUTREG01template.nii.gz'



#~~~~~~~~~~~~~~~~~~~~~below are subject- and study-specific events.~~~~~~~~~~~~~~~~~~~~~#

def get_events(subject_id):
	
	#i took the 'look' events out b/c they seemed to cause problems when only one event was present in the .txt file
	
	#event_1_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/CI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	#event_5_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/FACE_pre.run001.txt' %{"MYSUBJECT":(subject_id)}
	#event_9_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_look.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_negative.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_positive.run001.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run1	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/001/HAI_pre.run001.txt' %{"MYSUBJECT":(subject_id)}

	#event_1_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/CI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	#event_5_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/FACE_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	#event_9_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_look.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_negative.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_positive.run002.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run2	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/002/HAI_pre.run002.txt' %{"MYSUBJECT":(subject_id)}
	
	#event_1_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/CI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	#event_5_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/FACE_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	#event_9_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_look.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_10_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_negative.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_11_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_positive.run003.txt' %{"MYSUBJECT":(subject_id)}
	event_12_run3	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/003/HAI_pre.run003.txt' %{"MYSUBJECT":(subject_id)}
	
	#event_1_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_look.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_2_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_negative.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_3_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_positive.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_4_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/CI_pre.run004.txt' %{"MYSUBJECT":(subject_id)}
	#event_5_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_look.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_6_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_negative.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_7_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_positive.run004.txt' %{"MYSUBJECT":(subject_id)}
	event_8_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/FACE_pre.run004.txt' %{"MYSUBJECT":(subject_id)}
	#event_9_run4	= '/Volumes/lashley/AutReg.01/Data/behav/%(MYSUBJECT)s/004/HAI_look.run004.txt' %{"MYSUBJECT":(subject_id)}
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


#define a few contrasts.		
contrast1 = ('facepre>anyreg','T', 	['FACE_pre','FACE_positive','FACE_negative'],		[1,-0.5,-0.5])
contrast2 = ('anyreg>facepre','T', 	['FACE_pre','FACE_positive','FACE_negative'],		[-1,0.5,0.5])
contrast3 = ('faceneg>facepos','T',	['FACE_negative','FACE_positive'],			[1,-1])
contrast4 = ('facepos>faceneg','T', 	['FACE_positive','FACE_negative'],			[1,-1])
contrast5 = ('facepre>facepos','T', 	['FACE_pre','FACE_positive'],				[1,-1])
contrast6 = ('facepos>facepre','T',	['FACE_positive','FACE_pre'],				[1,-1])

			
contrasts = [contrast1,contrast2,contrast3,contrast4,contrast5,contrast6]
#,contrast7,contrast8,contrast9,contrast10,contrast11,contrast12,contrast13,contrast14,contrast15,contrast16,contrast17,contrast18,contrast19,contrast20,contrast21,contrast22,contrast23,contrast24,contrast25,contrast26,contrast27,contrast28,contrast29,contrast30,contrast31,contrast32,contrast33,contrast34,contrast35,contrast36,contrast37,contrast38,contrast39]


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
#~~~~~~~~~~~~~~~~~~~~~below are necessary functions~~~~~~~~~~~~~~~~~~~~~#

def pathfinder(subject, T1name):
	import os
	experiment_dir = '/Volumes/lashley/AutReg.01/Data/anat/%s/anat.nii.gz' (subject)
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


datasource = pe.Node(

		interface=nio.DataGrabber(
		
			base_directory			= subjects_dir,
			template 			= '/Data/anat/%s/' + T1_identifier,
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
datagrabber.inputs.field_template 			= dict	(func = experiment_dir + '/Data/func/%s/%s/f.nii.gz',
								 anat = experiment_dir + '/Data/anat/%s/anat.nii.gz',
								 )
								 
datagrabber.inputs.sorted 				= True
datagrabber.inputs.run					= runs


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
			#time_acquisition		= ((study_TR)-(study_TR/32)), 	#these are spm specific
			#slice_order			= range(32,0,-1),		#these are spm specific
			#ref_slice 			= 1,				#these are spm specific
			#num_slices 			= 32),				#these are spm specific
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

flirt = pe.MapNode(
	
		interface=fsl.FLIRT(
		
			cost 				= 'mutualinfo',
			cost_func 			= 'mutualinfo',
			dof 				= 12,
			verbose 			= 2),
			
				iterfield		= ['in_file'],
				name			= 'flirt')

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
			
				iterfield 		= ['realigned_files', 'realignment_parameters'],
				name			= "art")


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
								'flirt_registered_file',
								'flirt_out_matrix_file',
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
preproc.connect(realign, 'realigned_files',maskfunc,'in_file')
preproc.connect(maskfunc, 'out_file', getthresh, 'in_file')
preproc.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')
preproc.connect(maskfunc, 'out_file', threshold, 'in_file')
preproc.connect(threshold, 'out_file', dilatemask, 'in_file')
preproc.connect(dilatemask, 'out_file', outputnode, 'mask')
preproc.connect(realign, 'realigned_files', maskfunc2, 'in_file')
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
preproc.connect(inputnode,'subject_id',func2anat,'subject_id')
preproc.connect(meanfunc,'out_file',outputnode,'mean_func')
preproc.connect(realign,'realigned_files',flirt,'in_file')			#flirt
preproc.connect(datagrabber,'anat',flirt,'reference')				#flirt
preproc.connect(flirt,'out_file',outputnode,'flirt_registered_file')		#flirt
preproc.connect(flirt,'out_matrix_file',outputnode,'flirt_out_matrix_file')	#flirt
preproc.connect(func2anat,'min_cost_file',outputnode,'min_cost_file')		#bbreg
preproc.connect(func2anat,'out_fsl_file',outputnode,'out_fsl_file')		#bbreg
preproc.connect(func2anat,'out_reg_file',outputnode,'out_reg_file')		#bbreg
preproc.connect(func2anat,'registered_file',outputnode,'registered_file')	#bbreg

preproc.connect(susansmooth,'outputnode.smoothed_files', outputnode,'smoothed_files')	#susan smooth -  main output from preproc


#if using highpass filter. 
#preproc.connect(inputnode, ('highpass', highpass_operand), highpass, 'op_string')
#preproc.connect(realign,'realigned_files',highpass,'in_file')


#if using rapid art
#preproc.connect(dilate,'out_file',art,'mask_file')				# art
#preproc.connect(realign,'realignment_parameters',art,'realignment_parameters')	# art
#preproc.connect(realign,'realigned_files',art,'realigned_files')		# art
#preproc.connect(art,'outlier_files',outputnode,'outlier_files')		# art
#preproc.connect(art,'intensity_files',outputnode,'intensity_files')		# art
#preproc.connect(art,'norm_files',outputnode,'norm_files')			# art
#preproc.connect(art,'plot_files',outputnode,'plot_files')			# art
#preproc.connect(art,'statistic_files',outputnode,'statistic_files')		# art




#~~~~~~~~~~~~~~~~~~~~~below is 1st level~~~~~~~~~~~~~~~~~~~~~#



#initiate the 1st level workflow
firstlevel_volume = pe.Workflow(
	
		base_dir 				= subjects_dir,
		name					= 'firstlevel_volume')


infosource = pe.Node(

		interface=util.IdentityInterface(
		
			fields				= ['subject_id']),
			iterables 			= ('subject_id', subjects_list),
			
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
			bases 				= {'hrf':{'derivs': [1,0]}} ),
			
				name			= "level1design")


level1estimate = pe.Node(

		interface=spm.EstimateModel(
		
			estimation_method 		= {'Classical' : 1},
			), name				= "level1estimate")
		

contrastestimate = pe.Node(

		interface = spm.EstimateContrast(
			
			), name				="contrastestimate")
		

#get the warp and afflines produced by ANTS
def get_transformation_series(subject_id):
	
	image		= '/Volumes/lashley/AutReg.01/Analysis/ANTS/AUTREG01%s_brainWarp.nii.gz' %(subject_id)
	affline 	= '/Volumes/lashley/AutReg.01/Analysis/ANTS/AUTREG01%s_brainAffine.txt' %(subject_id)
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
		item = filename[0] 
		
		print '==========================\n'
		print item
		print '==========================\n'
		return item
	except:
        	print '==========================\n'
		print "Can't pick registration2anat"
		print '==========================\n'
		

				
vol2vol = pe.MapNode(
		
		interface = fs.ApplyVolTransform(
		
			no_resample 			= True),
			iterfield			= ['source_file'],
			name 				= "vol2vol")
			
'''

#these are optional, and slow.  Will produce png files depicting thumbnail sketches of overlays
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
'''	
		
firstlevel_outputnode = pe.Node(

		interface=util.IdentityInterface(
			
			fields				=['session_info',
							'mask_image',
							'in_anat',
							'bbreg_files',
							'flirt_out_matrix_file',
							'resliced_pngs',
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
firstlevel_volume.connect(vol2vol,'transformed_file',warp,'moving_image')
firstlevel_volume.connect(warp,'output_images',firstlevel_outputnode,'warped2template') 		#outputs from 1stlevel
firstlevel_volume.connect(vol2vol,'transformed_file',firstlevel_outputnode,'in_anat')			#outputs from 1stlevel
firstlevel_volume.connect(inputnode,'bbreg_files',firstlevel_outputnode,'bbreg_files')

#uncomment to create overlay images in datasink
#firstlevel_volume.connect(warp,'output_images',selectcontrast,'inlist')
#firstlevel_volume.connect(selectcontrast,'out',overlaystats,'stat_image')
#firstlevel_volume.connect(overlaystats,'out_file',slicestats,'in_file')
#firstlevel_volume.connect(slicestats,'out_file',firstlevel_outputnode,'resliced_pngs')


#Clone the vol analysis and prep for surface analysis
firstlevel_surf = firstlevel_volume.clone(name='firstlevel_surface')


frameflow = pe.Workflow(

			base_dir 			= subjects_dir,
			name				= 'workflows')

frameflow.connect(preproc,('outputnode.anat'),firstlevel_volume,('vol2vol.target_file'))
frameflow.connect(preproc,('outputnode.aseg_mask'),firstlevel_volume,('level1design.mask_image'))
frameflow.connect(preproc,('outputnode.flirt_out_matrix_file',pickBBReg),firstlevel_volume,('firstlevel_outputnode.flirt_out_matrix_file'))

frameflow.connect(preproc,('outputnode.flirt_out_matrix_file',pickBBReg),firstlevel_volume,('vol2vol.fsl_reg_file'))
frameflow.connect(preproc,('outputnode.subject_id',get_events),firstlevel_volume,'modelspec.event_files')
frameflow.connect(preproc,('outputnode.subject_id',get_transformation_series),firstlevel_volume,'warp.transformation_series')
frameflow.connect([(
	
			preproc,firstlevel_volume,[
			
						('outputnode.smoothed_files','convert_preproc2nii.in_file'),
						],

						) ])

#the connections for surf analysis
frameflow.connect(preproc,('outputnode.anat'),firstlevel_surf,('vol2vol.target_file'))
frameflow.connect(preproc,('outputnode.flirt_out_matrix_file',pickBBReg),firstlevel_surf,('vol2vol.fsl_reg_file'))


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
						('firstlevel_volume.firstlevel_outputnode.flirt_out_matrix_file','2nd_level_VOL_results.@reg'),
						('firstlevel_volume.firstlevel_outputnode.resliced_pngs','2nd_level_VOL_results.@pngs'),
						

						('firstlevel_surface.firstlevel_outputnode.in_anat','2nd_level_SURF_results.@in_anat'),
						('firstlevel_volume.firstlevel_outputnode.flirt_out_matrix_file','2nd_level_SURF_results.@flirt'),
						('firstlevel_surface.firstlevel_outputnode.resliced_pngs','2nd_level_SURF_results.@pngs'),
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